// Extend React's JSX namespace for <model-viewer> custom element
declare namespace React.JSX {
  interface IntrinsicElements {
    "model-viewer": React.DetailedHTMLProps<
      React.HTMLAttributes<HTMLElement> & {
        src?: string;
        alt?: string;
        "camera-controls"?: boolean | string;
        "auto-rotate"?: boolean | string;
        "rotation-per-second"?: string;
        "shadow-intensity"?: string;
        "shadow-softness"?: string;
        "environment-image"?: string;
        exposure?: string;
        "camera-orbit"?: string;
        "interaction-prompt"?: string;
        "touch-action"?: string;
        ar?: boolean | string;
        "ar-modes"?: string;
      },
      HTMLElement
    >;
  }
}
