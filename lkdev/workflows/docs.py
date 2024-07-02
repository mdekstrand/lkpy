from ..ghactions import script
from ._common import PACKAGES, step_checkout

PYTHON_VERSION = "3.11"


def workflow():
    return {
        "name": "Automatic Tests",
        "on": {
            "push": {
                "branches": ["main"],
            },
            "pull_request": {},
        },
        "defaults": {
            "run": {
                "shell": "bash -el {0}",
            },
        },
        "concurrency": {
            "group": "test-${{github.ref}}",
            "cancel-in-progress": True,
        },
        "jobs": {"build": job_build_docs()},
    }


def job_build_docs():
    return {
        "name": "Build documentation",
        "runs-on": "ubuntu-lates",
        "stages": stages_setup() + stages_build_doc() + stages_package(),
    }


def stages_setup():
    pip = ["pip install --no-deps"]
    pip += [f"-e {pkg}" for pkg in PACKAGES]
    return [
        step_checkout(),
        {
            "id": "setup-env",
            "name": "📦 Set up Conda environment",
            "uses": "mamba-org/setup-micromamba@v1",
            "with": {
                "environment-file": "docs/environment.yml",
                "environment-name": "lkpy",
                "init-shell": "bash",
            },
        },
        {
            "id": "install",
            "name": "🍱 Install LensKit packages",
            "run": script.command(pip),
        },
    ]


def stages_build_doc():
    return [
        {
            "id": "docs",
            "name": "📚 Build documentation site",
            "run": script.command(["just docs"]),
        }
    ]


def stages_package():
    return [
        {
            "name": "📤 Package documentation site",
            "uses": "actions/upload-artifact@v4",
            "with": {
                "name": "lenskit-docs",
                "path": "build/doc",
            },
        }
    ]


def job_publish_docs():
    return {
        "name": "Publish documentation",
        "runs-on": "ubuntu-latest",
        "needs": ["build"],
        "environment": "docs",
        "steps": [
            {
                "id": "decrypt",
                "name": "🔓 Decrypt deployment key",
                "run": script("""
                    tmpdir=$(mktemp -d lksite.XXXXXX)
                    echo "$AGE_DECRYPT" >$tmpdir/decrypt-identity
                    echo 'deploy-key<<EOK' >>$GITHUB_OUTPUT
                    age -d -i $tmpdir/decrypt-identity etc/doc-deploy-key.asc >>$GITHUB_OUTPUT
                    echo 'EOK' >>$GITHUB_OUTPUT
                    rm -rf $tmpdir
                """),
                "env": {
                    "AGE_DECRYPT": "${{ secrets.DOC_DEPLOY_DECRYPT_KEY }}",
                },
            },
            {
                "name": "Check out doc site",
                "uses": "actions/checkout@v4",
                "with": {
                    "repository": "lenskit/lenskit-docs",
                    "ssh-key": "${{steps.decrypt.output.decrypt-identity}}",
                    "path": "doc-site",
                    "ref": "latest",
                },
            },
            {
                "name": "📥 Fetch documentation package",
                "uses": "actions/download-artifact@v4",
                "with": {
                    "name": "lenskit-docs",
                    "path": "build/doc",
                },
            },
            {
                "name": "🛻 Copy documentation content",
                "run": script("""
                    rsync -av --delete --exclude=.git/ build/doc/ doc-site/
                    cd doc-site
                    git config user.name "LensKit Doc Bot"
                    git config user.email "docbot@lenskit.org"
                    git add .
                    git commit -m 'rebuild documentation'
                    git push
                """),
            },
        ],
    }
