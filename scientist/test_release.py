import copy
import hashlib
import io
import json
import runpy
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import release

ROOT = Path(__file__).resolve().parent.parent
TAG = "v0.6.2-sci.12"
VERSION = "0.6.2+sci.12"
COMMIT = "a" * 40
RUN_URL = "https://github.com/scientist-labs/scispacy/actions/runs/123"


def wheel_bytes(name, version, requirements=()):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as wheel:
        metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
        metadata += "".join(f"Requires-Dist: {item}\n" for item in requirements)
        wheel.writestr(f"{name}-{version}.dist-info/METADATA", metadata)
    return buffer.getvalue()


def package(directory, requirements=("nmslib>=2.1.2", "typer<0.26")):
    (directory / f"scispacy-{VERSION}-py3-none-any.whl").write_bytes(wheel_bytes("scispacy", VERSION, requirements))
    with tarfile.open(directory / f"scispacy-{VERSION}.tar.gz", "w:gz") as source:
        data = f"Name: scispacy\nVersion: {VERSION}\n".encode()
        member = tarfile.TarInfo(f"scispacy-{VERSION}/PKG-INFO")
        member.size = len(data)
        source.addfile(member, io.BytesIO(data))


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.assets = self.root / "assets"
        self.assets.mkdir()
        self.manifest = json.loads((ROOT / "scientist/nmslib-wheels.json").read_text())

    def test_first_release_ignores_upstream_tags(self):
        self.assertEqual(release.next_release("0.6.2", ["v0.6.2", "v0.5.5"]), ("v0.6.2-sci.1", "0.6.2+sci.1"))

    def test_counter_is_numeric_and_global_across_upstream_versions(self):
        self.assertEqual(release.next_release("0.7.0", ["v0.6.2-sci.9", "v0.6.2-sci.12", "v0.7.0-sci.2"]), ("v0.7.0-sci.13", "0.7.0+sci.13"))

    def test_noncanonical_tags_fail_closed(self):
        for tag in ["v0.6.2-sci.0", "v0.6.2-sci.01", "v0.6.2-sci.x", "v0.6.2-sci.3\n"]:
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                release.next_release("0.6.2", [tag])
        with self.assertRaises(ValueError):
            release.next_release("0.6.2+sci.1", [])

    def test_stamp_preserves_upstream_components_and_is_not_repeatable(self):
        path = self.root / "version.py"
        path.write_text((ROOT / "scispacy/version.py").read_text())
        release.stamp(path, TAG)
        values = runpy.run_path(str(path))
        self.assertEqual(values["VERSION"], VERSION)
        self.assertEqual(values["VERSION_SHORT"], "0.6")
        with self.assertRaises(ValueError):
            release.stamp(path, TAG)

    def test_stamp_rejects_different_upstream_version(self):
        path = self.root / "version.py"
        path.write_text('VERSION = "0.6.2"\n')
        with self.assertRaises(ValueError):
            release.stamp(path, "v0.7.0-sci.12")

    def test_package_identity_and_dependency(self):
        package(self.assets)
        release.verify_package(self.assets, TAG)
        with self.assertRaises(ValueError):
            release.verify_package(self.assets, "v0.6.2-sci.13")
        (self.assets / "unexpected.whl").touch()
        with self.assertRaises(ValueError):
            release.verify_package(self.assets, TAG)

    def test_package_rejects_metabrainz_and_missing_official_dependency(self):
        for requirements in [("nmslib-metabrainz==2.1.3",), ("nmslib>=2.1.2", "nmslib-metabrainz==2.1.3"), ()]:
            with self.subTest(requirements=requirements):
                package(self.assets, requirements)
                with self.assertRaises(ValueError):
                    release.verify_package(self.assets, TAG)

    def test_package_rejects_missing_spacy_cli_compatibility_bound(self):
        package(self.assets, ("nmslib>=2.1.2",))
        with self.assertRaisesRegex(ValueError, "spaCy/Click"):
            release.verify_package(self.assets, TAG)

    def test_reviewed_manifest_has_both_native_cp311_wheels(self):
        parsed = release.read_manifest(ROOT / "scientist/nmslib-wheels.json")
        self.assertEqual(set(parsed), {"amd64", "arm64"})
        for item in parsed.values():
            self.assertIn("cp311-cp311-manylinux2014", item["filename"])

    def write_manifest(self):
        path = self.root / "manifest.json"
        path.write_text(json.dumps(self.manifest))
        return path

    def test_manifest_rejects_other_hosts_platforms_or_missing_arch(self):
        original = copy.deepcopy(self.manifest)
        for field, value in [("url", "https://example.com/wheel.whl"), ("filename", "../wheel.whl"), ("sha256", "wrong")]:
            self.manifest = copy.deepcopy(original)
            self.manifest["arm64"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                release.read_manifest(self.write_manifest())
        self.manifest.pop("amd64")
        with self.assertRaises(ValueError):
            release.read_manifest(self.write_manifest())

    def populate_native_assets(self):
        data = wheel_bytes("nmslib", "2.1.2")
        for arch, item in self.manifest.items():
            item["sha256"] = hashlib.sha256(data).hexdigest()
            item["size"] = len(data)
            (self.assets / item["filename"]).write_bytes(data)
            proof = {"arch": arch, "version": VERSION, "commit": COMMIT, "status": "passed"}
            (self.assets / f"proof-linux-{arch}-cp311.json").write_text(json.dumps(proof))
        return data

    def test_download_checks_hash_and_removes_partial_on_failure(self):
        data = self.populate_native_assets()
        with patch("urllib.request.urlopen", return_value=io.BytesIO(data)):
            release.fetch_wheel(self.write_manifest(), "arm64", self.assets)
        destination = self.assets / self.manifest["arm64"]["filename"]
        self.assertEqual(destination.read_bytes(), data)
        destination.unlink()
        with patch("urllib.request.urlopen", return_value=io.BytesIO(b"corrupt")), self.assertRaises(ValueError):
            release.fetch_wheel(self.write_manifest(), "arm64", self.assets)
        self.assertFalse(destination.exists())
        self.assertFalse(destination.with_suffix(".partial").exists())

    def test_bundle_requires_both_architectures_and_writes_checksums(self):
        package(self.assets)
        self.populate_native_assets()
        release.bundle(self.assets, self.write_manifest(), TAG, COMMIT, RUN_URL)
        entries = (self.assets / "SHA256SUMS").read_text().splitlines()
        self.assertEqual(len(entries), 7)
        for entry in entries:
            digest, filename = entry.split("  ", 1)
            self.assertEqual(digest, release.sha256(self.assets / filename))
        provenance = json.loads((self.assets / "PROVENANCE.json").read_text())
        self.assertEqual(provenance["commit"], COMMIT)
        self.assertEqual(set(provenance["third_party_wheels"]), {"arm64", "amd64"})

    def test_bundle_rejects_missing_proof_or_wrong_commit(self):
        package(self.assets)
        self.populate_native_assets()
        proof = self.assets / "proof-linux-arm64-cp311.json"
        proof.unlink()
        with self.assertRaises(FileNotFoundError):
            release.bundle(self.assets, self.write_manifest(), TAG, COMMIT, RUN_URL)
        self.populate_native_assets()
        with self.assertRaises(ValueError):
            release.bundle(self.assets, self.write_manifest(), TAG, "b" * 40, RUN_URL)

    def test_bundle_rejects_corrupt_native_wheel(self):
        package(self.assets)
        self.populate_native_assets()
        (self.assets / self.manifest["arm64"]["filename"]).write_bytes(b"corrupt")
        with self.assertRaises(ValueError):
            release.bundle(self.assets, self.write_manifest(), TAG, COMMIT, RUN_URL)


if __name__ == "__main__":
    unittest.main()
