import argparse

from config.settings import DEFAULT_INPUT_VIDEO
from pipeline.run_full_analysis import run_full_analysis


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT_VIDEO)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-video", default=None)
    parser.add_argument("--metadata-file", default=None)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--resize-width", type=int, default=None)
    parser.add_argument("--face-model", choices=["hog", "cnn"], default=None)
    parser.add_argument("--upsample", type=int, default=None)
    parser.add_argument("--face-fallback", choices=["haar", "none"], default=None)
    parser.add_argument("--haar-scale", type=float, default=None)
    parser.add_argument("--haar-neighbors", type=int, default=None)
    parser.add_argument("--min-face-size", type=int, default=None)
    parser.add_argument("--face-padding", type=float, default=None)
    parser.add_argument("--full-metadata", action="store_true")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    run_full_analysis(
        args.input,
        output_dir=args.output_dir,
        output_video=args.output_video,
        metadata_file=args.metadata_file,
        frame_step=args.frame_step,
        max_frames=args.max_frames,
        resize_width=args.resize_width,
        face_model=args.face_model,
        upsample=args.upsample,
        face_fallback=args.face_fallback,
        haar_scale=args.haar_scale,
        haar_neighbors=args.haar_neighbors,
        summary_only=not args.full_metadata,
        min_face_size=args.min_face_size,
        face_padding=args.face_padding,
    )


if __name__ == "__main__":
    main()
