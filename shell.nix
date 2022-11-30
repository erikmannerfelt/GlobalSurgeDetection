{ pkgs ? import <nixpkgs> { } }:
let
  fhs = pkgs.buildFHSUserEnv rec {
    name = "surgedetection";

    targetPkgs = _: with pkgs; [
      micromamba
      which
      libGL
      zsh
    ];

    profile = ''
      set -e
      eval "$(micromamba shell hook -s bash | grep -v 'complete -o')"

      export MAMBA_ROOT_PREFIX="$(pwd)/.mamba"

      if ! [[ -d "$MAMBA_ROOT_PREFIX/envs" ]]; then
        mkdir -p $MAMBA_ROOT_PREFIX/envs
      fi
      # If an environment with the same name exists, don't create a new one
      ls $MAMBA_ROOT_PREFIX/envs | grep ${name} || micromamba create -q -n ${name}

      # Activate the environment
      micromamba activate ${name}

      micromamba install --offline -f environment.yml -y
      set +e
      zsh
    '';
  };
in
fhs.env
