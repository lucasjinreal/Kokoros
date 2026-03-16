{
  description = "Kokoros - Rust TTS workspace";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };

        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" ];
        };

        nativeBuildInputs = with pkgs; [
          rustToolchain
          pkg-config
          cmake
        ];

        buildInputs = with pkgs; [
          # espeak-ng for espeak-rs
          espeak-ng

          # Audio encoding libraries
          libopus
          libogg
          lame

          # ONNX Runtime
          onnxruntime

          # OpenSSL for reqwest
          openssl
        ] ++ lib.optionals stdenv.isDarwin [
          apple-sdk_15
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          inherit nativeBuildInputs buildInputs;

          env = {
            RUST_BACKTRACE = "1";
            PKG_CONFIG_PATH = pkgs.lib.makeSearchPath "lib/pkgconfig" buildInputs;
            ORT_STRATEGY = "system";
            ORT_LIB_LOCATION = "${pkgs.onnxruntime}/lib";
          };

          shellHook = ''
            echo "Kokoros development shell"
            echo "Rust: $(rustc --version)"
          '';
        };
      }
    );
}
