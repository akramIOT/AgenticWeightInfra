import tempfile
import unittest
from pathlib import Path

from wsp.model import make_tiny_qwen
from wsp.report import edit_report
from wsp.runtime import run_program
from wsp.workspace import ControlWorkspace
from wsp.__main__ import main


class ReportTest(unittest.TestCase):
    def test_skip_zeros_the_touched_tensors_and_leaves_the_neighbor_out(self):
        info = make_tiny_qwen()
        program = "layers.10-12 skip"
        result = run_program(info, program, operator="RAILS", reason="middle layer surgery")
        rows = edit_report(info, result["build"].patched, result["compiled"])
        names = {row["name"] for row in rows}
        self.assertIn("model.layers.10.mlp.down_proj.weight", names)
        self.assertNotIn("model.layers.9.mlp.down_proj.weight", names)
        zeroed = next(row for row in rows if row["name"].endswith("layers.10.mlp.down_proj.weight"))
        self.assertEqual(zeroed["cosine"], 0.0)
        self.assertGreater(zeroed["maxAbs"], 0.0)
        self.assertEqual(zeroed["normRatio"], 0.0)

    def test_scale_keeps_direction_and_moves_the_norm(self):
        info = make_tiny_qwen()
        program = "layers.10.mlp.down_proj scale 1.3"
        result = run_program(info, program, operator="RAILS", reason="scale the mlp")
        row = edit_report(info, result["build"].patched, result["compiled"])[0]
        self.assertGreater(row["cosine"], 0.999)
        self.assertAlmostEqual(row["normRatio"], 1.3, places=4)


class WorkspaceTest(unittest.TestCase):
    def test_controlled_apply_waits_for_a_second_person_and_keeps_the_chain(self):
        info = make_tiny_qwen()
        program = "layers.10-12 skip"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = ControlWorkspace(root, environment="controlled")
            blocked = workspace.run(info, program, operator="ada", reason="middle layer surgery")
            self.assertFalse(blocked["applied"])
            self.assertIn("second person", blocked["gate"]["blockReason"])
            self.assertEqual(len(workspace.audit.list()), 1)

            explained = workspace.explain(info, program)
            self.assertEqual(explained["policy"]["decision"], "allow")
            workspace.save_program(
                name="skip-middle",
                model_id=info.rec["model_id"],
                program=program,
                author="ada",
                decision=explained["policy"]["decision"],
            )
            with self.assertRaisesRegex(ValueError, "different person"):
                workspace.approve("skip-middle", "ada")
            workspace.approve("skip-middle", "bao")

            reopened = ControlWorkspace(root)
            self.assertEqual(reopened.environment, "controlled")
            opened = reopened.run(info, program, operator="ada", reason="middle layer surgery")
            self.assertTrue(opened["applied"])
            self.assertTrue(reopened.audit.verify()["ok"])
            self.assertGreaterEqual(len(reopened.audit.list()), 2)

    def test_cli_explain_report_and_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            code = main(["explain", "--program", "layers.4-6 skip", "--environment", "lab"])
            self.assertIsNone(code)
            code = main(["report", "--program", "layers.10.mlp.down_proj scale 1.3"])
            self.assertIsNone(code)
            code = main([
                "workspace", "--dir", directory, "--environment", "lab",
                "save", "--name", "skip-middle", "--program", "layers.10-12 skip", "--operator", "ada",
            ])
            self.assertIsNone(code)
            code = main(["workspace", "--dir", directory, "approve", "--name", "skip-middle", "--approver", "bao"])
            self.assertIsNone(code)
            code = main(["workspace", "--dir", directory, "verify"])
            self.assertIsNone(code)
            listed = ControlWorkspace(directory).registry.list()
            self.assertEqual(listed[0]["status"], "approved")


if __name__ == "__main__":
    unittest.main()
