{
  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    base_env.url = "path:../base_env";
    base_env.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, flake-utils, base_env, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        python = base_env.packages.${system}.python_from_requirements ./requirements.txt;

        packages = pkgs.lib.attrsets.recursiveUpdate
          (builtins.listToAttrs (map (pkg: { name = pkg.pname; value = pkg; }) (with pkgs; [
            pre-commit
            zsh
            google-cloud-sdk
          ])))
          {
            inherit python;

          };

      in
      {
        inherit packages;
        defaultPackage = packages.python;

        devShell = pkgs.mkShell {
          name = "EarthEngineGlobalSurgeDetection";
          buildInputs = pkgs.lib.attrValues packages;
          shellHook = ''

            zsh
          '';
        };
      }

    );
}
