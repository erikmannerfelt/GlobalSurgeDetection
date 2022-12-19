{ pkgs ? import <nixpkgs> { } }:
let
  fhs = pkgs.buildFHSUserEnv rec {
    name = "surgedetection";

    targetPkgs = _: with pkgs; [
      micromamba
      which
      libGL
      zsh
      yq
      jq
      openssh
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

      # The micromamba solver is slow, so here's a faster check to see if all packages are installed
      # Note that version bumps are not respected!

      # List the dependencies in the environment.yml
      specified_packages=`cat environment.yml | yq '.dependencies[]' -r | sed -e 's/<.*//g' -e 's/>.*//g' -e 's/=.*//g'`
      # List the installed packages
      installed_packages=`micromamba list --json | jq '.[].name' -r`

      # Count the list of specified dependencies
      n_specified_packages=`echo $specified_packages | wc -w`
      # Out of the specified packages, count how many are installed
      n_installed=`echo "$specified_packages $installed_packages" | tr ' ' '\n' | sort | uniq -d | wc -l`

      # If these are not equal, something has changed and this is run again
      if [[ "$n_installed" != "$n_specified_packages" ]]; then
        micromamba install -f environment.yml -y
      fi
      set +e
      zsh
    '';
  };
in
fhs.env
