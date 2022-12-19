{
  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        packages = builtins.listToAttrs (map (pkg: { name = pkg.pname; value = pkg; }) ([
          (import ./python.nix { inherit pkgs; })
          pkgs.pre-commit
          pkgs.zsh
        ]));

      in
      {
        inherit packages;
        defaultPackage = packages.python;

        devShell = pkgs.mkShell {
          name = "GlobalSurgeDetection";
          buildInputs = pkgs.lib.attrValues packages;
          shellHook = ''

            zsh
          '';
        };
      }

    );
}
