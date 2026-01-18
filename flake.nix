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
          config.allowUnfree = true; # Required for CUDA
        };

        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" ];
        };

        nativeBuildInputs = with pkgs; [
          rustToolchain
          pkg-config
          cmake
          llvmPackages.libclang
          gcc
          alsa-utils
        ];

        commonBuildInputs = with pkgs; [
          # espeak-ng for espeak-rs
          espeak-ng

          # Audio encoding libraries
          libopus
          libogg
          lame

          # OpenSSL for reqwest
          openssl
        ] ++ lib.optionals stdenv.isDarwin [
          apple-sdk_15
        ];

        commonEnv = {
          RUST_BACKTRACE = "1";
          PKG_CONFIG_PATH = pkgs.lib.makeSearchPath "lib/pkgconfig" commonBuildInputs;
          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
          BINDGEN_EXTRA_CLANG_ARGS = builtins.concatStringsSep " " [
            "-I${pkgs.llvmPackages.libclang.lib}/lib/clang/${pkgs.llvmPackages.libclang.version}/include"
            "-I${pkgs.glibc.dev}/include"
          ];
        };
      in
      {
        devShells.default = pkgs.mkShell {
          inherit nativeBuildInputs;
          buildInputs = commonBuildInputs;

          env = commonEnv // {
            # Use download strategy since nixpkgs onnxruntime (1.22.2) is too old for ort 2.0.0-rc.11 (needs 1.23+)
            ORT_STRATEGY = "download";
            # Set RPATH to $ORIGIN so binaries can find ONNX Runtime libraries in the same directory
            CARGO_BUILD_RUSTFLAGS = "-C link-arg=-Wl,-rpath,$ORIGIN";
          };

          shellHook = ''
            echo "Kokoros development shell"
            echo "Rust: $(rustc --version)"
          '';
        };

        devShells.cuda = pkgs.mkShell {
          inherit nativeBuildInputs;
          buildInputs = commonBuildInputs;

          env = commonEnv // {
            # Use download strategy since nixpkgs onnxruntime (1.22.2) is too old for ort 2.0.0-rc.11 (needs 1.23+)
            ORT_STRATEGY = "download";
            # Set RPATH to $ORIGIN so binaries can find ONNX Runtime libraries in the same directory
            CARGO_BUILD_RUSTFLAGS = "-C link-arg=-Wl,-rpath,$ORIGIN";
            # Add CUDA to library path for runtime
            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
              pkgs.cudaPackages.cudatoolkit
              pkgs.cudaPackages.cudnn
            ];
          };

          shellHook = ''
            echo "Kokoros development shell (CUDA enabled)"
            echo "Rust: $(rustc --version)"
            echo "CUDA: $(nvcc --version 2>/dev/null || echo 'not in PATH')"
          '';
        };
      }
    );
}
