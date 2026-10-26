"""PTX emission regression check; run with make test-ptx CUDA_ARCH=sm_100a.

This checks the descriptor fence's scope, size and operand in every compiled
TMA specialization. It is not a general control-flow or runtime ordering proof.
"""
import os
from pathlib import Path
import re
import unittest


@unittest.skipUnless(os.environ.get("BENCHMARK_PTX"), "requires emitted PTX (make test-ptx)")
class TmaDescriptorPtxTests(unittest.TestCase):
    def test_each_tma_kernel_acquires_its_global_descriptor(self):
        ptx = Path(os.environ["BENCHMARK_PTX"]).read_text()
        target = re.search(r"\.target\s+sm_(\d+)(a?)", ptx)
        self.assertIsNotNone(target)
        arch = int(target[1])
        expected = 12 if arch == 100 and target[2] == "a" else (3 if arch >= 90 else 0)
        entries = re.split(r"\.visible\s+\.entry\s+", ptx)[1:]
        entries = [entry for entry in entries if "cp.async.bulk.tensor." in entry]
        self.assertEqual(len(entries), expected)
        for entry in entries:
            with self.subTest(kernel=entry.split("(", 1)[0]):
                fences = list(re.finditer(
                    r"fence\.proxy\.tensormap::generic\.acquire\.sys\s*"
                    r"\[([^\]]+)\],\s*128\s*;", entry))
                self.assertEqual(len(fences), 1, "need one system-scope descriptor acquire per CTA")
                copies = list(re.finditer(
                    r"cp\.async\.bulk\.tensor\.[^;]+?\[[^\]]+\],\s*\[([^,]+),", entry))
                self.assertTrue(copies)
                # nvcc may reload the immutable pointer parameter after the
                # inline asm's memory clobber, using a different PTX register.
                param_loads = dict(re.findall(
                    r"ld\.param\.u64\s+(%\w+),\s*\[([^\]]+)\]", entry))
                def origin(register):
                    register = register.strip()
                    return param_loads.get(register, register)
                for copy in copies:
                    self.assertEqual(origin(fences[0][1]), origin(copy[1]),
                                     "acquire must cover the descriptor used by TMA")
                # The acquire is in the initial issuer block, before prefetch.
                self.assertLess(fences[0].start(), copies[0].start())
