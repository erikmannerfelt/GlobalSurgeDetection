{
  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    base_env.url = "./modules/base_env";
    base_env.inputs.nixpkgs.follows = "nixpkgs";
    base_env.inputs.flake-utils.follows = "flake-utils";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, base_env, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = base_env.devShell.${system};
      }

    );
}
