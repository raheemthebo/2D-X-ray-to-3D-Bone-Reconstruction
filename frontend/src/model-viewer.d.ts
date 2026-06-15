// Extend React's JSX namespace for <model-viewer> custom element
declare namespace React.JSX {
  interface IntrinsicElements {
    "model-viewer": React.DetailedHTMLProps<
      React.HTMLAttributes<HTMLElement> & {
        src?: string;
        alt?: string;
        "camera-controls"?: boolean | string;
        "auto-rotate"?: boolean | string;
        "shadow-intensity"?: string;
        "environment-image"?: string;
        exposure?: string;
      },
      HTMLElement
    >;
  }
}
