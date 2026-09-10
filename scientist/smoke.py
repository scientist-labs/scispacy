"""Exercise the installed wheel, not the source checkout, without model downloads."""

import argparse
import importlib.metadata
import json
import platform
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version")
    parser.add_argument("arch", choices=["amd64", "arm64"])
    parser.add_argument("commit")
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    assert sys.version_info[:2] == (3, 11), sys.version
    assert platform.system() == "Linux", platform.system()
    assert platform.machine() == {"amd64": "x86_64", "arm64": "aarch64"}[args.arch]
    assert importlib.metadata.version("scispacy") == args.version
    assert importlib.metadata.version("nmslib") == "2.1.2"
    try:
        importlib.metadata.version("nmslib-metabrainz")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        raise AssertionError("both ANN distributions were installed")

    # A local-only KB must not silently fall back to upstream models or datasets.
    with patch("socket.socket.connect", side_effect=AssertionError("unexpected network access")):
        import numpy
        import scipy.sparse
        import scispacy.version
        import spacy
        from scispacy.candidate_generation import (
            CandidateGenerator, LinkerPaths, create_tfidf_ann_index,
        )
        from scispacy.linking_utils import KnowledgeBase
        from spacy.cli._util import app
        from typer.testing import CliRunner

        assert scispacy.version.VERSION == args.version, "imported the unversioned source checkout"
        assert Path(scispacy.version.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
        kb = KnowledgeBase(args.fixture)
        with tempfile.TemporaryDirectory() as temporary:
            aliases, vectorizer, index = create_tfidf_ann_index(temporary, kb)
            assert len(aliases) == len(index) == 93
            vectors = scipy.sparse.load_npz(Path(temporary) / "tfidf_vectors_sparse.npz")
            assert vectors.dtype == numpy.float32, "lost upstream's newer-SciPy serialization fix"
            # Reload the serialized index through scispaCy's real load path.
            aliases, vectorizer, index = LinkerPaths.from_directory(temporary).load()
            generator = CandidateGenerator(index, vectorizer, aliases, kb)
            candidates = generator(["(131)I-Macroaggregated Albumin"], 10)
            exact = [item for item in candidates[0] if item.concept_id == "C0000005"]
            assert exact and max(exact[0].similarities) > 0.999
            assert generator(["ZZZZ"], 10) == [[]]
            assert generator([], 10) == []
        assert spacy.blank("en")("biomedical text")[0].text == "biomedical"
        cli_result = CliRunner().invoke(app, ["info"])
        assert cli_result.exit_code == 0, (cli_result.output, cli_result.exception)

    proof = {
        "status": "passed", "version": args.version, "arch": args.arch,
        "commit": args.commit, "python": platform.python_version(),
        "platform": platform.platform(), "scispacy_import": scispacy.version.VERSION,
        "checks": ["wheel identity", "native ANN import", "TF-IDF float32 serialization",
                   "ANN save/load/query", "empty/OOV queries", "spaCy blank pipeline",
                   "spaCy CLI info", "no network during smoke"],
        "installed": dict(sorted((distribution.metadata["Name"], distribution.version)
                                 for distribution in importlib.metadata.distributions())),
    }
    args.output.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
    print(f"Native {args.arch} wheel smoke passed for {args.version}")


if __name__ == "__main__":
    main()
