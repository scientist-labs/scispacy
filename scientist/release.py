"""Scientist release metadata and verified third-party wheel downloads (stdlib only)."""

import argparse
import hashlib
import json
import re
import runpy
import subprocess
import tarfile
import urllib.request
import zipfile
from email.parser import BytesParser
from pathlib import Path

REPOSITORY = "scientist-labs/scispacy"
UPSTREAM_COMMIT = "eacccd4ef3e7ef13d4aa35700e718bd4318ded17"
TAG_PATTERN = re.compile(r"v(\d+\.\d+\.\d+)-sci\.([1-9]\d*)\Z")
BASE_PATTERN = re.compile(r"\d+\.\d+\.\d+\Z")


def next_release(base, tags):
    if not BASE_PATTERN.fullmatch(base):
        raise ValueError(f"expected an unmodified upstream X.Y.Z version, got {base!r}")
    numbers = []
    for tag in tags:
        match = TAG_PATTERN.fullmatch(tag)
        if match:
            numbers.append(int(match[2]))
        elif "-sci." in tag:
            raise ValueError(f"noncanonical Scientist tag: {tag!r}")
    number = max(numbers, default=0) + 1
    return f"v{base}-sci.{number}", f"{base}+sci.{number}"


def package_version(tag):
    match = TAG_PATTERN.fullmatch(tag)
    if not match:
        raise ValueError(f"invalid Scientist release tag: {tag!r}")
    return f"{match[1]}+sci.{match[2]}"


def stamp(version_file, tag):
    version = runpy.run_path(str(version_file))["VERSION"]
    match = TAG_PATTERN.fullmatch(tag)
    if not match or match[1] != version:
        raise ValueError("tag base must match the unmodified upstream version")
    with version_file.open("a") as stream:
        stream.write(f'\n# Scientist release identity; upstream version components stay intact.\nVERSION += "+sci.{match[2]}"\n')


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def wheel_metadata(path):
    with zipfile.ZipFile(path) as wheel:
        names = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
        if len(names) != 1:
            raise ValueError(f"expected exactly one wheel METADATA: {path.name}")
        return BytesParser().parsebytes(wheel.read(names[0]))


def verify_package(directory, tag):
    version = package_version(tag)
    expected_wheel = directory / f"scispacy-{version}-py3-none-any.whl"
    expected_sdist = directory / f"scispacy-{version}.tar.gz"
    if sorted(path.name for path in directory.iterdir()) != sorted([expected_wheel.name, expected_sdist.name]):
        raise ValueError("package directory must contain exactly the versioned universal wheel and sdist")
    metadata = wheel_metadata(expected_wheel)
    if metadata["Name"] != "scispacy" or metadata["Version"] != version:
        raise ValueError("wheel metadata does not match its release identity")
    requirements = metadata.get_all("Requires-Dist", [])
    if "nmslib>=2.1.2" not in requirements or any("metabrainz" in item for item in requirements):
        raise ValueError("wheel lost the upstream #559 dependency fix")
    if "typer<0.26" not in requirements:
        raise ValueError("wheel lost the spaCy/Click compatibility bound")
    with tarfile.open(expected_sdist, "r:gz") as source:
        member = source.extractfile(f"scispacy-{version}/PKG-INFO")
        if member is None or BytesParser().parsebytes(member.read())["Version"] != version:
            raise ValueError("sdist version does not match the wheel")


def read_manifest(path):
    manifest = json.loads(path.read_text())
    if set(manifest) != {"arm64", "amd64"}:
        raise ValueError("exactly arm64 and amd64 third-party wheels are required")
    for arch, item in manifest.items():
        platform = {"arm64": "aarch64", "amd64": "x86_64"}[arch]
        expected = f"nmslib-2.1.2-cp311-cp311-manylinux2014_{platform}.manylinux_2_17_{platform}.whl"
        if item["filename"] != expected:
            raise ValueError("unexpected third-party wheel name/platform")
        if not item["url"].startswith("https://files.pythonhosted.org/packages/") or not item["url"].endswith("/" + expected):
            raise ValueError("third-party source must be the reviewed PyPI artifact")
        if not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
            raise ValueError("invalid third-party artifact SHA256")
    return manifest


def verify_third_party(path, item):
    if path.stat().st_size != item["size"] or sha256(path) != item["sha256"]:
        raise ValueError(f"third-party wheel failed size/hash verification: {path.name}")
    metadata = wheel_metadata(path)
    if metadata["Name"] != "nmslib" or metadata["Version"] != "2.1.2":
        raise ValueError("unexpected third-party distribution metadata")


def fetch_wheel(manifest_path, arch, directory):
    item = read_manifest(manifest_path)[arch]
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / item["filename"]
    temporary = destination.with_suffix(".partial")
    try:
        with urllib.request.urlopen(item["url"], timeout=60) as response:
            # The exact expected size bounds both memory use and corrupt downloads.
            data = response.read(item["size"] + 1)
        temporary.write_bytes(data)
        verify_third_party(temporary, item)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def bundle(directory, manifest_path, tag, commit, run_url):
    version = package_version(tag)
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("expected an exact source commit")
    if not re.fullmatch(r"https://github.com/scientist-labs/scispacy/actions/runs/\d+", run_url):
        raise ValueError("expected this fork's Actions run URL")
    manifest = read_manifest(manifest_path)
    expected = {f"scispacy-{version}-py3-none-any.whl", f"scispacy-{version}.tar.gz"}
    for arch, item in manifest.items():
        verify_third_party(directory / item["filename"], item)
        expected.add(item["filename"])
        proof_name = f"proof-linux-{arch}-cp311.json"
        proof = json.loads((directory / proof_name).read_text())
        if (proof["arch"], proof["version"], proof["commit"], proof["status"]) != (arch, version, commit, "passed"):
            raise ValueError(f"proof identity mismatch: {proof_name}")
        expected.add(proof_name)
    if {path.name for path in directory.iterdir()} != expected:
        raise ValueError("release assets are missing, duplicated, or unexpected")
    files = {name: {"sha256": sha256(directory / name), "size": (directory / name).stat().st_size} for name in sorted(expected)}
    provenance = {
        "repository": REPOSITORY, "tag": tag, "version": version,
        "commit": commit, "upstream_baseline": UPSTREAM_COMMIT, "run_url": run_url,
        "third_party_wheels": manifest, "files": files,
        "scope": "Linux CPython 3.11 wheel installation and local ANN smoke tests; not model/production certification",
    }
    (directory / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    checksums = "".join(f"{sha256(path)}  {path.name}\n" for path in sorted(directory.iterdir()))
    (directory / "SHA256SUMS").write_text(checksums)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--version-file", type=Path, default=Path("scispacy/version.py"))
    stamping = commands.add_parser("stamp")
    stamping.add_argument("tag")
    stamping.add_argument("--version-file", type=Path, default=Path("scispacy/version.py"))
    verify = commands.add_parser("verify-package")
    verify.add_argument("tag")
    verify.add_argument("directory", type=Path)
    download = commands.add_parser("fetch-nmslib")
    download.add_argument("arch", choices=["arm64", "amd64"])
    download.add_argument("directory", type=Path)
    download.add_argument("--manifest", type=Path, default=Path("scientist/nmslib-wheels.json"))
    manifest = commands.add_parser("bundle")
    manifest.add_argument("tag")
    manifest.add_argument("commit")
    manifest.add_argument("run_url")
    manifest.add_argument("directory", type=Path)
    manifest.add_argument("--manifest", type=Path, default=Path("scientist/nmslib-wheels.json"))
    args = parser.parse_args()
    if args.command == "plan":
        base = runpy.run_path(str(args.version_file))["VERSION"]
        tags = subprocess.check_output(["git", "tag", "--list"], text=True).splitlines()
        tag, version = next_release(base, tags)
        print(f"tag={tag}\nversion={version}")
    elif args.command == "stamp":
        stamp(args.version_file, args.tag)
    elif args.command == "verify-package":
        verify_package(args.directory, args.tag)
    elif args.command == "fetch-nmslib":
        fetch_wheel(args.manifest, args.arch, args.directory)
    else:
        bundle(args.directory, args.manifest, args.tag, args.commit, args.run_url)


if __name__ == "__main__":
    main()
