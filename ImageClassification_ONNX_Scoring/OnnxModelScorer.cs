using ImageClassification_ONNX_Scoring.Helper;
using ImageClassification_ONNX_Scoring.Model;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using static ImageClassification_ONNX_Scoring.Helper.ConsoleHelpers;
using static ImageClassification_ONNX_Scoring.Helper.ModelHelpers;

namespace ImageClassification_ONNX_Scoring
{
    class OnnxModelScorer
    {
        private readonly string dataLocation;
        private readonly string imagesFolder;
        private readonly string modelLocation;
        private readonly string labelsLocation;

        private readonly MLContext mlContext;

        public OnnxModelScorer(string dataLocation, string imagesFolder, string modelLocation, string labelsLocation)
        {
            this.dataLocation = dataLocation;
            this.imagesFolder = imagesFolder;
            this.modelLocation = modelLocation;
            this.labelsLocation = labelsLocation;
            mlContext = new MLContext();
        }

        public struct ImageNetSettings
        {
            public const int imageHeight = 224;
            public const int imageWidth = 224;
        }

        public struct ResNet152v2ModelSettings
        {
            // input tensor name
            public const string ModelInput = "data";

            // output tensor name
            public const string ModelOutput = "resnetv27_dense0_fwd";
        }

        private ITransformer LoadModel(string dataLocation, string imagesFolder, string modelLocation)
        {
            Console.WriteLine("Read model");
            Console.WriteLine($"Model location: {modelLocation}");
            Console.WriteLine($"Default parameters: image size=({ImageNetSettings.imageWidth},{ImageNetSettings.imageHeight})");

            // Create IDataView from empty list to obtain input data schema
            var data = mlContext.Data.LoadFromEnumerable(new List<ImageNetData>());

            IEstimator<ITransformer> pipeline =
                mlContext.Transforms.LoadImages(
                    outputColumnName: ResNet152v2ModelSettings.ModelInput,
                    imageFolder: imagesFolder,
                    inputColumnName: nameof(ImageNetData.ImagePath))
                            // The image transforms transform the images into the model's expected format.
                            .Append(mlContext.Transforms.ResizeImages(
                                outputColumnName: ResNet152v2ModelSettings.ModelInput,
                                imageWidth: ImageNetSettings.imageWidth,
                                imageHeight: ImageNetSettings.imageHeight,
                                inputColumnName: ResNet152v2ModelSettings.ModelInput))
                            .Append(mlContext.Transforms.ExtractPixels(
                                outputColumnName: ResNet152v2ModelSettings.ModelInput))
                            // The ScoreTensorFlowModel transform scores the TensorFlow model and allows communication 
                            .Append(mlContext.Transforms.ApplyOnnxModel(
                                modelFile: modelLocation,
                                outputColumnNames: new[] { ResNet152v2ModelSettings.ModelOutput },
                                inputColumnNames: new[] { ResNet152v2ModelSettings.ModelInput }
                            ));

            ITransformer model = pipeline.Fit(data);

            IDataView testData = mlContext.Data.LoadFromTextFile<ImageNetData>(path: dataLocation, hasHeader: false);
            IDataView predictions = model.Transform(testData);
            // Create an IEnumerable for the predictions for displaying results
            IEnumerable<ImageNetPrediction> imagePredictionData = mlContext.Data.CreateEnumerable<ImageNetPrediction>(predictions, true);

            DisplayResults(imagePredictionData);

            return model;
        }

        private static void DisplayResults(IEnumerable<ImageNetPrediction> imagePredictionData)
        {
            foreach (ImageNetPrediction prediction in imagePredictionData)
            {
                Console.WriteLine($"Image score: {prediction.PredictedLabels.Max()} ");
            }
        }

        public void ClassifySingleImage(string imagePath, ITransformer model)
        {
            Console.WriteLine("=============== Making single image classification ===============");
            // load the fully qualified image file name into ImageData 
            var imageData = new ImageNetData()
            {
                ImagePath = imagePath
            };
            var labels = ModelHelpers.ReadLabels(labelsLocation);
            // Make prediction function (input = ImageData, output = ImagePrediction)
            var predictor = mlContext.Model.CreatePredictionEngine<ImageNetData, ImageNetPrediction>(model);
            var probs = predictor.Predict(imageData).PredictedLabels;

            var imageDataP = new ImageNetDataProbability()
            {
                ImagePath = imagePath,
                Label = ""
            };
            (imageDataP.PredictedLabel, imageDataP.Probability) = GetBestLabel(labels, probs);
            imageDataP.ConsoleWrite();
        }

        public void PredictImage(string testLocation, string imagesFolder, string labelsLocation, ITransformer model)
        {
            ConsoleWriteHeader("Classificate images");
            Console.WriteLine($"Images folder: {imagesFolder}");
            Console.WriteLine($"Training file: {testLocation}");
            Console.WriteLine($"Labels file: {labelsLocation}");

            var labels = ModelHelpers.ReadLabels(labelsLocation);

            var testData = ImageNetData.ReadFromCsv(testLocation, imagesFolder);

            var predictor = mlContext.Model.CreatePredictionEngine<ImageNetData, ImageNetPrediction>(model);

            foreach (var sample in testData)
            {
                var probs = predictor.Predict(sample).PredictedLabels;
                var abc = Softmax(probs);
                var imageData = new ImageNetDataProbability()
                {
                    ImagePath = sample.ImagePath,
                    Label = sample.Label
                };
                (imageData.PredictedLabel, imageData.Probability) = GetBestLabel(labels, abc);
                imageData.ConsoleWrite();
            }
        }

        public void Score()
        {
            var model = LoadModel(dataLocation, imagesFolder, modelLocation);

            PredictImage(dataLocation, imagesFolder, labelsLocation, model);
        }

        private float[] Softmax(float[] values)
        {
            var maxVal = values.Max();
            var exp = values.Select(v => Math.Exp(v - maxVal));
            var sumExp = exp.Sum();

            return exp.Select(v => (float)(v / sumExp)).ToArray();
        }
    }
}
