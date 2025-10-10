# Firefly V2 Nodes

This folder contains refactored versions of the Adobe Firefly API nodes with improved structure, organization, and maintainability.

## What's New in V2?

### Improved Code Organization
- **Modular Structure**: All nodes are organized in a dedicated package
- **Cleaner Code**: Reduced boilerplate and better separation of concerns
- **Helper Functions**: Shared utility functions for common operations
- **Better Documentation**: Enhanced docstrings and inline comments

### Enhanced Functionality
- **Simplified Parameters**: Cleaner parameter handling with better defaults
- **Better Error Handling**: Improved error messages and validation
- **Enhanced Debug Logging**: More informative console output
- **Batch Processing**: Optimized batch processing for all nodes

### Key Differences from V1

| Feature | V1 | V2 |
|---------|----|----|
| Code Organization | Single large file | Modular package |
| Debug Logging | Basic console output | Enhanced formatted logs |
| Error Handling | Basic exceptions | Detailed error messages |
| Parameter Organization | Mixed order | Grouped by category |
| Documentation | Minimal | Comprehensive |

## Available Nodes

All V2 nodes are available in the **"api node/firefly v2"** category:

### 1. Upload Image V2
- Upload images to Firefly storage
- Returns upload ID for use in other nodes
- **Node ID**: `FireflyUploadImageNodeV2`

### 2. Text to Image V2
- Generate images from text prompts
- Supports multiple model versions (image3, image4_standard, image4_ultra, etc.)
- Style and structure reference support
- **Node ID**: `FireflyTextToImageNodeV2`

### 3. Generative Fill V2
- Fill/inpaint masked areas of images
- Optional prompt guidance
- **Node ID**: `FireflyGenerativeFillNodeV2`

### 4. Generative Expand V2
- Expand/outpaint images to larger sizes
- Optional prompt guidance
- **Node ID**: `FireflyGenerativeExpandNodeV2`

### 5. Generate Similar V2
- Generate similar images based on reference
- Optional prompt guidance
- **Node ID**: `FireflyGenerateSimilarNodeV2`

### 6. Object Composite V2
- Generate and composite objects into scenes
- Requires background image and placement mask
- **Node ID**: `FireflyGenerateObjectCompositeNodeV2`

## Usage Examples

### Basic Text to Image
1. Add **Text to Image V2** node
2. Enter your prompt
3. Select model version (default: image4_standard)
4. Choose size/aspect ratio
5. Execute

### With Style Reference
1. Add **Upload Image V2** node with style reference image
2. Add **Text to Image V2** node
3. Connect upload_id to style_upload_id
4. Set style_strength (0-100)
5. Execute

### Generative Fill
1. Add **Generative Fill V2** node
2. Connect image and mask inputs
3. Optionally add a prompt
4. Execute

## Migration from V1 to V2

### Node Name Changes
- `FireflyUploadImageNode` → `FireflyUploadImageNodeV2`
- `FireflyTextToImageNode` → `FireflyTextToImageNodeV2`
- `FireflyGenerativeFillNode` → `FireflyGenerativeFillNodeV2`
- `FireflyGenerativeExpandNode` → `FireflyGenerativeExpandNodeV2`
- `FireflyGenerateSimilarNode` → `FireflyGenerateSimilarNodeV2`
- `FireflyGenerateObjectCompositeNode` → `FireflyGenerateObjectCompositeNodeV2`

### Category Changes
- V1 Category: `"api node/image/firefly"`
- V2 Category: `"api node/firefly v2"`

### Parameter Changes
All parameters remain the same, but V2 has:
- Better parameter grouping in the UI
- Improved tooltips and documentation
- Better default values

## Backward Compatibility

- **V1 nodes remain unchanged** - all existing workflows continue to work
- **V2 nodes are additions** - use them in new workflows or migrate gradually
- **Shared backend** - both versions use the same Firefly API infrastructure

## Technical Details

### Dependencies
- Same as V1 nodes (Adobe Firefly API access required)
- Requires `firefly_config.json` with OAuth credentials

### File Structure
```
comfy_api_nodes/
├── firefly_v2/
│   ├── __init__.py           # Package initialization and node registration
│   ├── firefly_easy_nodes.py # All V2 node implementations
│   └── README.md             # This file
```

### EasyNodes Integration
While this package initially explored using ComfyUI-EasyNodes for simplified node creation, we found that the async nature of the Firefly API and complex optional parameters were better served by traditional node implementation with improved structure. The "V2" name reflects the improved architecture rather than a specific framework dependency.

## Future Improvements

Planned enhancements for future versions:
- [ ] Additional debug output options
- [ ] Preset configurations for common use cases
- [ ] Enhanced batch processing options
- [ ] Style/structure preset library
- [ ] Video generation support (when API available)

## Support

For issues or questions:
1. Check the main Firefly API documentation
2. Verify your `firefly_config.json` credentials
3. Check the ComfyUI console for detailed error messages
4. Ensure all dependencies are installed

## License

Same as main ComfyUI project.
