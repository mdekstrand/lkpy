import { WorkflowStep } from "@lenskit/typeline/github";

import { pythonVersionString, TestJobSpec } from "./spec.ts";
import { script } from "../lib/script.ts";

export interface VanillaTestOpts extends TestJobSpec {
  install: "vanilla";
  req_files?: string[];
  dep_strategy?: "minimum" | "default";
  pip_args?: string[];
}

export function isVanillaSpec(spec: TestJobSpec): spec is VanillaTestOpts {
  return spec.install == "vanilla";
}

export function vanillaSetup(options: VanillaTestOpts): WorkflowStep[] {
  let pip = "uv pip install --python $PYTHON";
  for (let pkg of requiredPackages(options)) {
    if (pkg == "lenskit") {
      pkg = "lenskit[test]";
    }
    pip += ` "./${pkg}"`;
  }
  if (options.dep_strategy == "minimum") {
    pip += " --resolution=lowest-direct";
  }
  if (options.pip_args) {
    pip += " " + options.pip_args.join(" ");
  }

  return [
    {
      name: "🐍 Set up Python",
      id: "install-python",
      uses: "actions/setup-python@v5",
      with: {
        "python-version": pythonVersionString(options),
      },
    },
    {
      name: "🕶️ Set up uv",
      run: script("pip install -U 'uv>=0.1.15'"),
    },
    {
      name: "📦 Set up Python dependencies",
      id: "install-deps",
      run: script(pip),
      shell: "bash",
      env: {
        PYTHON: "${{steps.install-python.outputs.python-path}}",
        UV_EXTRA_INDEX_URL: "https://download.pytorch.org/whl/cpu",
        UV_INDEX_STRATEGY: "unsafe-first-match",
      },
    },
  ];
}

function requiredPackages(options: VanillaTestOpts) {
  let pkgs = options.packages ?? [];
  if (!pkgs.includes("lenskit")) {
    pkgs = ["lenskit", ...pkgs];
  }
  return pkgs;
}
