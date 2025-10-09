/**
 * Firefly Dual Input Extension
 *
 * Enables dual input mode for Firefly nodes:
 * - Connection inputs show red dot when disconnected, green when connected
 * - Click input socket to show/hide text field below it
 * - Text field appears inline and pushes other widgets down
 * - Values from connection and text field are concatenated
 */

import { app } from "../../scripts/app.js";

const DUAL_INPUT_PARAMS = [
    "negative_prompt",
    "custom_model_id",
    "prompt_biasing_locale",
    "style_upload_id",
    "style_presets",
    "structure_upload_id",
    "prompt_suffix"
];

app.registerExtension({
    name: "Comfy.FireflyDualInput",

    async setup() {
        console.log("[Firefly] Dual input extension loaded");
    },

    async nodeCreated(node) {
        // Only process Firefly nodes with dual input support
        if (!node.type || !node.type.startsWith("Firefly")) {
            return;
        }

        console.log(`[Firefly] Processing node: ${node.type}`);

        // Store reference to text widgets for each dual input
        node.fireflyTextWidgets = {};
        node.fireflyTextWidgetVisibility = {};

        // Find and hide text companion widgets
        if (node.widgets) {
            for (let i = 0; i < node.widgets.length; i++) {
                const widget = node.widgets[i];

                // Check if this is a text companion widget
                if (widget.name && widget.name.endsWith("_text")) {
                    const baseName = widget.name.replace("_text", "");

                    if (DUAL_INPUT_PARAMS.includes(baseName)) {
                        console.log(`[Firefly] Found text widget: ${widget.name}`);

                        // Store reference
                        node.fireflyTextWidgets[baseName] = widget;
                        node.fireflyTextWidgetVisibility[baseName] = false;

                        // Hide by default
                        widget.type = "hidden";
                        Object.defineProperty(widget, 'computeSize', {
                            value: () => [0, -4], // Negative height to hide completely
                            writable: true,
                            configurable: true
                        });
                    }
                }
            }
        }

        // Add click handlers to input sockets
        if (node.inputs) {
            for (let i = 0; i < node.inputs.length; i++) {
                const input = node.inputs[i];

                if (DUAL_INPUT_PARAMS.includes(input.name)) {
                    console.log(`[Firefly] Adding click handler to input: ${input.name}`);

                    // Store original onMouseDown if it exists
                    const originalOnMouseDown = node.onMouseDown;

                    // Override onMouseDown to detect clicks on inputs
                    node.onMouseDown = function(e, localPos, graphCanvas) {
                        // Check if click is on this input socket
                        const inputY = this.pos[1] + (i + 1) * 26; // Approximate Y position
                        const inputX = this.pos[0];

                        // Check if click is near the input socket (20px radius)
                        const dist = Math.sqrt(
                            Math.pow(e.canvasX - inputX, 2) +
                            Math.pow(e.canvasY - inputY, 2)
                        );

                        if (dist < 20) {
                            // Toggle text widget visibility
                            const textWidget = this.fireflyTextWidgets[input.name];
                            if (textWidget) {
                                const isVisible = this.fireflyTextWidgetVisibility[input.name];

                                if (isVisible) {
                                    // Hide text widget
                                    textWidget.type = "hidden";
                                    Object.defineProperty(textWidget, 'computeSize', {
                                        value: () => [0, -4],
                                        writable: true,
                                        configurable: true
                                    });
                                    this.fireflyTextWidgetVisibility[input.name] = false;
                                } else {
                                    // Show text widget
                                    textWidget.type = input.name.includes("prompt") ? "text" : "string";
                                    delete textWidget.computeSize; // Restore default size computation
                                    this.fireflyTextWidgetVisibility[input.name] = true;
                                }

                                // Force node size recalculation
                                this.setSize(this.computeSize());
                                graphCanvas.setDirty(true);

                                console.log(`[Firefly] Toggled ${input.name} text widget:`, !isVisible);
                                return true; // Prevent default behavior
                            }
                        }

                        // Call original handler if it exists
                        if (originalOnMouseDown) {
                            return originalOnMouseDown.call(this, e, localPos, graphCanvas);
                        }
                    };
                }
            }
        }

        // Override onConnectionsChange to update socket colors
        const originalOnConnectionsChange = node.onConnectionsChange;
        node.onConnectionsChange = function(type, index, connected, link_info) {
            if (type === 1) { // Input connection
                const input = this.inputs[index];

                if (input && DUAL_INPUT_PARAMS.includes(input.name)) {
                    // Update socket color based on connection state
                    if (connected) {
                        input.color_on = "#0F0"; // Green when connected
                        console.log(`[Firefly] Input ${input.name} connected - GREEN`);
                    } else {
                        input.color_on = "#F00"; // Red when disconnected
                        console.log(`[Firefly] Input ${input.name} disconnected - RED`);
                    }
                }
            }

            // Call original handler
            if (originalOnConnectionsChange) {
                originalOnConnectionsChange.call(this, type, index, connected, link_info);
            }
        };

        // Set initial socket colors (red for disconnected)
        if (node.inputs) {
            for (let i = 0; i < node.inputs.length; i++) {
                const input = node.inputs[i];
                if (DUAL_INPUT_PARAMS.includes(input.name)) {
                    input.color_on = "#F00"; // Red by default
                }
            }
        }

        console.log(`[Firefly] Node ${node.type} dual input setup complete`);
    }
});
