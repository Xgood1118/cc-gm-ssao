import argparse
import ssao as m


def main():
    ap = argparse.ArgumentParser(description="SSAO 核采样与降噪")
        ap.add_argument("--radius", type=float, default=4.0)
        ap.add_argument("--samples", type=int, default=8)
        ap.add_argument("--out", required=True)
    a = ap.parse_args()
        print("TODO")


if __name__ == "__main__":
    main()
