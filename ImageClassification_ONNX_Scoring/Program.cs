using ImageClassification_ONNX_Scoring.Model;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Linq;

namespace ImageClassification_ONNX_Scoring
{
    class Program
    {
        public static void Main()
        {
            var assetsRelativePath = @"assets";
            string assetsPath = GetAbsolutePath(assetsRelativePath);
            var tagsTsv = Path.Combine(assetsPath, "images", "tags.tsv");
            var modelFilePath = Path.Combine(assetsPath, "resnet152v2", "resnet152v2.onnx");
            var labelsTxt = Path.Combine(assetsPath, "resnet152v2", "synset_text.txt");

            var imagesFolder = Path.Combine(assetsPath, "images");
            var outputFolder = Path.Combine(assetsPath, "images", "output");

            // Initialize MLContext
            MLContext mlContext = new MLContext();

            try
            {
                // Load Data
                IEnumerable<ImageNetData> images = ImageNetData.ReadFromFile(imagesFolder);
                IDataView imageDataView = mlContext.Data.LoadFromEnumerable(images);

                // Create instance of model scorer
                var modelScorer = new OnnxModelScorer(tagsTsv, imagesFolder, modelFilePath, labelsTxt);

                // Use model to score data
                modelScorer.Score();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }

            Console.WriteLine("========= End of Process..Hit any Key ========");
            Console.ReadLine();
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }
    }
}
