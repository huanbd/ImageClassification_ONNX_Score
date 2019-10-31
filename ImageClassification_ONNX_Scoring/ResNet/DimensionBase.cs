using System;
using System.Collections.Generic;
using System.Text;

namespace ImageClassification_ONNX_Scoring.ResNet
{
    public class DimensionBase
    {
        public float X { get; set; }
        public float Y { get; set; }
        public float Height { get; set; }
        public float Width { get; set; }
    }
}
