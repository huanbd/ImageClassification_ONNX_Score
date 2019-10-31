using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace ImageClassification_ONNX_Scoring.Model
{
    public class ImageNetPrediction
    {
        [ColumnName(OnnxModelScorer.ResNet152v2ModelSettings.ModelOutput)]
        public float[] PredictedLabels;
    }
}
