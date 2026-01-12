# syntax=docker/dockerfile:1
FROM rust:1.88.0-slim-trixie AS builderrs

RUN apt-get update -qq && apt-get install -qq -y wget pkg-config libssl-dev clang git cmake && rustup component add rustfmt

WORKDIR /app

COPY . .
COPY Cargo.toml .
COPY Cargo.lock .

RUN chmod +x ./download_all.sh && ./download_all.sh

RUN cargo build --release

FROM debian:sid-slim AS runner

WORKDIR /app

COPY --from=builderrs /app/target/release/build ./target/release/build
COPY --from=builderrs /app/target/release/koko ./target/release/koko
COPY --from=builderrs /app/data ./data
COPY --from=builderrs /app/checkpoints ./checkpoints

RUN chmod +x ./target/release/koko && apt-get update -qq && apt-get install -qq -y pkg-config libssl-dev 

EXPOSE 3000

ENTRYPOINT [ "./target/release/koko" ]
