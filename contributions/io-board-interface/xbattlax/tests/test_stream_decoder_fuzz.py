"""Property/fuzz tests for StreamDecoder against untrusted UART bytes.

The serial link between the CPU and the STM32 I/O board carries safety-relevant
traffic over a noisy physical medium (line errors, resets, partial writes).
StreamDecoder must therefore stay robust no matter what byte soup it is fed:
never crash, never emit a malformed frame, never grow its buffer without bound,
and produce the same frames regardless of how the stream is chunked.

These tests are pure stdlib ``unittest`` with a fixed RNG seed, so they are
deterministic and are auto-discovered by the existing ``python`` CI job with no
third-party dependencies.
"""

import pathlib
import random
import struct
import sys
import unittest


TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from oomwoo_mcu_frame import (  # noqa: E402
    CRC_SIZE,
    HEADER_FORMAT,
    HEADER_SIZE,
    MAGIC,
    MAX_PAYLOAD_SIZE,
    VERSION,
    FrameDecodeError,
    StreamDecoder,
    decode_frame,
    encode_frame,
)

MAX_FRAME_SIZE = HEADER_SIZE + MAX_PAYLOAD_SIZE + CRC_SIZE  # 524

# A few fixed seeds so a flake, if any, is reproducible and we cover more space.
SEEDS = (1, 7, 42, 1337, 90210)


def _random_valid_frame(rng, max_payload=64):
    """Encode one valid frame with random-but-legal fields.

    Payloads are kept small by default to keep the pure-Python CRC cost low;
    the 0- and 512-byte boundaries are covered explicitly in the targeted test.
    """
    payload_len = rng.randint(0, max_payload)
    payload = bytes(rng.getrandbits(8) for _ in range(payload_len))
    return encode_frame(
        rng.randint(0, 0xFFFF),          # message_type
        payload,
        sequence=rng.randint(0, 0xFFFF),
        flags=rng.randint(0, 0xFF),
    )


def _biased_noise(rng, length):
    """Random bytes over-weighted toward the magic bytes to exercise sync."""
    pool = bytes(MAGIC) * 8 + bytes(range(256))
    return bytes(rng.choice(pool) for _ in range(length))


def _hexdump(data, limit=64):
    head = bytes(data[:limit]).hex()
    return head + ("..." if len(data) > limit else "")


class StreamDecoderFuzzTest(unittest.TestCase):
    def test_feed_never_raises_and_emits_only_valid_frames(self):
        """Arbitrary bytes must never crash feed(); emitted frames round-trip
        and the post-call buffer stays below one max frame."""
        for seed in SEEDS:
            rng = random.Random(seed)
            decoder = StreamDecoder()
            for iteration in range(2000):
                chunk = _biased_noise(rng, rng.randint(0, 64))
                try:
                    frames = decoder.feed(chunk)
                except Exception as exc:  # noqa: BLE001 - robustness is the point
                    self.fail(
                        f"feed() raised {exc!r} (seed={seed}, iter={iteration}, "
                        f"chunk={_hexdump(chunk)})"
                    )
                for frame in frames:
                    reencoded = encode_frame(
                        frame.message_type,
                        frame.payload,
                        sequence=frame.sequence,
                        flags=frame.flags,
                        version=frame.version,
                    )
                    self.assertEqual(
                        decode_frame(reencoded),
                        frame,
                        msg=f"emitted frame did not round-trip (seed={seed}, "
                        f"iter={iteration})",
                    )
                # Post-call, only an incomplete frame prefix may remain.
                self.assertLessEqual(
                    len(decoder._buffer),
                    MAX_FRAME_SIZE - 1,
                    msg=f"buffer grew to {len(decoder._buffer)} "
                    f"(seed={seed}, iter={iteration})",
                )

    def test_chunk_boundary_invariance(self):
        """Frames + residual buffer depend only on the concatenated stream,
        not on how it is split across feed() calls."""
        for seed in SEEDS:
            rng = random.Random(seed)
            for trial in range(200):
                parts = []
                for _ in range(rng.randint(0, 6)):
                    if rng.random() < 0.6:
                        parts.append(_random_valid_frame(rng))
                    else:
                        parts.append(_biased_noise(rng, rng.randint(0, 40)))
                stream = b"".join(parts)

                one_shot = StreamDecoder()
                expected = one_shot.feed(stream)

                chunked = StreamDecoder()
                got = []
                pos = 0
                while pos < len(stream):
                    step = rng.randint(1, 17)
                    got.extend(chunked.feed(stream[pos:pos + step]))
                    pos += step

                self.assertEqual(
                    got,
                    expected,
                    msg=f"chunking changed frame list (seed={seed}, trial={trial}, "
                    f"stream={_hexdump(stream)})",
                )
                self.assertEqual(
                    bytes(chunked._buffer),
                    bytes(one_shot._buffer),
                    msg=f"chunking changed residual buffer (seed={seed}, "
                    f"trial={trial})",
                )

    def test_valid_frames_recovered_through_non_magic_noise(self):
        """Valid frames separated by noise that cannot start a frame (no magic
        byte 'O') are all recovered, in order, exactly."""
        magic0 = MAGIC[0]
        for seed in SEEDS:
            rng = random.Random(seed)
            for trial in range(200):
                frames_bytes = [_random_valid_frame(rng) for _ in range(rng.randint(1, 5))]
                expected = [decode_frame(fb) for fb in frames_bytes]

                def noise():
                    n = rng.randint(0, 24)
                    return bytes(b for b in (rng.randint(0, 255) for _ in range(n)) if b != magic0)

                stream = noise()
                for fb in frames_bytes:
                    stream += fb + noise()

                got = StreamDecoder().feed(stream)
                self.assertEqual(
                    got,
                    expected,
                    msg=f"recovery failed (seed={seed}, trial={trial}, "
                    f"stream={_hexdump(stream)})",
                )


class StreamDecoderTargetedTest(unittest.TestCase):
    def test_oversized_declared_length_is_skipped(self):
        # A header claiming an illegal payload_len must be resynced past.
        bad_header = struct.pack(HEADER_FORMAT, MAGIC, VERSION, 0, 0, 0, MAX_PAYLOAD_SIZE + 1)
        good = encode_frame(0x0001, b"\xaa\xbb")
        got = StreamDecoder().feed(bad_header + good)
        self.assertEqual(got, [decode_frame(good)])

    def test_bad_crc_then_valid_frame(self):
        good = encode_frame(0x0001, b"payload")
        corrupt = bytearray(encode_frame(0x0002, b"zzz"))
        corrupt[-1] ^= 0xFF  # break the CRC
        got = StreamDecoder().feed(bytes(corrupt) + good)
        self.assertEqual(got, [decode_frame(good)])

    def test_empty_and_max_payload_round_trip(self):
        for payload_len in (0, MAX_PAYLOAD_SIZE):
            frame = encode_frame(0x0101, bytes(payload_len))
            got = StreamDecoder().feed(frame)
            self.assertEqual(got, [decode_frame(frame)])

    def test_max_payload_fed_in_fragments_stays_bounded(self):
        # A full-size frame delivered one byte at a time keeps the residual
        # buffer at the 523-byte bound right up until the final byte completes
        # and emits the frame.
        frame = encode_frame(0x0101, bytes(MAX_PAYLOAD_SIZE))
        self.assertEqual(len(frame), MAX_FRAME_SIZE)
        decoder = StreamDecoder()
        for byte in frame[:-1]:
            self.assertEqual(decoder.feed(bytes([byte])), [])
            self.assertLessEqual(len(decoder._buffer), MAX_FRAME_SIZE - 1)
        self.assertEqual(decoder.feed(frame[-1:]), [decode_frame(frame)])
        self.assertEqual(len(decoder._buffer), 0)

    def test_stalled_header_documents_known_limitation(self):
        # Known behavior (verified): a noise-borne header declaring the max
        # payload stalls the decoder behind an incomplete candidate, hiding a
        # trailing valid frame until enough bytes arrive to resync.
        fake_header = struct.pack(HEADER_FORMAT, MAGIC, VERSION, 0, 0, 0, MAX_PAYLOAD_SIZE)
        good = encode_frame(0x0001, b"")  # 12-byte frame
        decoder = StreamDecoder()
        self.assertEqual(decoder.feed(fake_header + good), [])
        self.assertEqual(len(decoder._buffer), len(fake_header) + len(good))
        # Recovery is only guaranteed because the completed max-length candidate
        # is itself invalid, so the decoder rejects it and resyncs to the real
        # frame. Assert that premise explicitly rather than relying on an
        # incidental CRC value of the filler bytes.
        filler = bytes(MAX_FRAME_SIZE)
        fake_candidate = (fake_header + good + filler)[:MAX_FRAME_SIZE]
        with self.assertRaises(FrameDecodeError):
            decode_frame(fake_candidate)
        # Padding to the declared size completes and rejects the fake candidate,
        # letting the resync advance and recover the real frame.
        recovered = decoder.feed(filler)
        self.assertEqual(recovered, [decode_frame(good)])


if __name__ == "__main__":
    unittest.main()
