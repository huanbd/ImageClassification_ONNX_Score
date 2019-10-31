using System;
using System.Collections.Generic;
using System.Drawing;
using System.Text;

namespace ImageClassification_ONNX_Scoring.ResNet
{
    public class BoundingBoxDimensions : DimensionBase { }
    public class BoundingBox
    {
        public BoundingBoxDimensions Dimensions { get; set; }

        public string Label { get; set; }

        public float Confidence { get; set; }

        public RectangleF Rect
        {
            get { return new RectangleF(Dimensions.X, Dimensions.Y, Dimensions.Width, Dimensions.Height); }
        }

        public Color BoxColor { get; set; }
    }
}
