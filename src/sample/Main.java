package sample;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;

import javafx.application.Application;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.VBox;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import static sample.Utils.mat2Image;

public class Main extends Application {
    @Override
    public void start(Stage stage) throws FileNotFoundException {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        //Zona de declaratii si initilizari

        //Initializam FieChooser
        FileChooser fileChooser = new FileChooser();
        File file = fileChooser.showOpenDialog(stage);

        //Creem un string care a salva path-ul imaginii selectate in FileChooser
        String path = file.getAbsolutePath();

        //Initializam clasa Imgproc, clasa care va face conversiile intre imagini.
        Imgcodecs imageCodecs = new Imgcodecs();

        //Creem un obiect de tip imagine care citeste path-ul nostru si preia informatia
        Image image = new Image(new FileInputStream(path));

        //Initializam clasa Mat care va crea o matrice in care vom stoca poza
        Mat negative = new Mat();
        Mat gray = new Mat();
        Mat albNegru = new Mat();
        Mat rgb = new Mat();
        Mat erode = new Mat();
        Mat delate = new Mat();

        //Initializam butoanele care vor afisa imaginile modificate
        Button Original = new Button("Imagine originala");
        Button Negative = new Button("Imagine negativata");
        Button Grayscale = new Button("Convert to Grayscale");
        Button Binary = new Button("Binary Color");
        Button hsv = new Button("RGB to HSV");
        Button histograma = new Button("Hisogram");
        Button fourier = new Button("Fourier transformation");
        Button eroziune = new Button("Erosion");
        Button dilatare = new Button("Dilatation");







            //Initializam imageView
            ImageView imageView = new ImageView();

            //Initializarea layout care contine butoane si ImageView in care va fi poza
            BorderPane Border = new BorderPane();
            Mat matrix = imageCodecs.imread(path);


            //Negativarea
            Mat invertcolormatrix = new Mat(matrix.rows(), matrix.cols(), matrix.type(), new Scalar(255, 255, 255));
            Core.subtract(invertcolormatrix, matrix, negative);

            //Grayscale
            Imgproc.cvtColor(matrix, gray, Imgproc.COLOR_RGB2GRAY);


            //Converting from grayscale to Binary Color
            Imgproc.threshold(gray, albNegru, 0, 255, Imgproc.THRESH_OTSU + Imgproc.THRESH_BINARY_INV);

            //Converting from RGB to HSV
            Imgproc.cvtColor(matrix, rgb, Imgproc.COLOR_RGB2HSV);

            //Histogram
            List<Mat> RGBPlanes = new ArrayList<>();
            Core.split(matrix, RGBPlanes);
            int marime = 256;
            /*Setarea range-ului*/
            float[] range = {0, 256};
            MatOfFloat MarimeHistograma = new MatOfFloat(range);
            boolean cumul = false;
            /*Trebuie sa despartim imaginea in planuri(rosu,verde si albastru)*/
            Mat HAlbastra = new Mat(), HVerde = new Mat(), HRosu = new Mat();

            Imgproc.calcHist(RGBPlanes, new MatOfInt(0), new Mat(),
                    HAlbastra, new MatOfInt(marime), MarimeHistograma, cumul);
            Imgproc.calcHist(RGBPlanes, new MatOfInt(1), new Mat(), HVerde,
                    new MatOfInt(marime), MarimeHistograma, cumul);
            Imgproc.calcHist(RGBPlanes, new MatOfInt(2), new Mat(), HRosu,
                    new MatOfInt(marime), MarimeHistograma, cumul);

            int histW = 512, histH = 400;/*Setarea latimii si inaltimii*/
            int binW = (int) Math.round((double) histW / marime);

            Mat histImage = new Mat(histH, histW, CvType.CV_8UC3, new Scalar(0, 0, 0));
            /*Normalizarea culorilor pentru a putea desena histograma*/
            Core.normalize(HAlbastra, HAlbastra, 0, histImage.rows(), Core.NORM_MINMAX);
            Core.normalize(HVerde, HVerde, 0, histImage.rows(), Core.NORM_MINMAX);
            Core.normalize(HRosu, HRosu, 0, histImage.rows(), Core.NORM_MINMAX);

            float[] bHistData = new float[(int) (HAlbastra.total() * HAlbastra.channels())];
            HAlbastra.get(0, 0, bHistData);
            float[] gHistData = new float[(int) (HVerde.total() * HVerde.channels())];
            HVerde.get(0, 0, gHistData);
            float[] rHistData = new float[(int) (HRosu.total() * HRosu.channels())];
            HRosu.get(0, 0, rHistData);
            /*Scalarea culorilor(Desenarea histogramei)*/
            for (int i = 1; i < marime; i++) {
                Imgproc.line(histImage, new Point(binW * (i - 1), histH - Math.round(bHistData[i - 1])),
                        new Point(binW * (i), histH - Math.round(bHistData[i])), new Scalar(255, 0, 0), 2);
                Imgproc.line(histImage, new Point(binW * (i - 1), histH - Math.round(gHistData[i - 1])),
                        new Point(binW * (i), histH - Math.round(gHistData[i])), new Scalar(0, 255, 0), 2);
                Imgproc.line(histImage, new Point(binW * (i - 1), histH - Math.round(rHistData[i - 1])),
                        new Point(binW * (i), histH - Math.round(rHistData[i])), new Scalar(0, 0, 255), 2);
            }

            //Fourier Transformation from a grayscale Image
            Mat Fourier = new Mat();
            Mat captusit = new Mat();
            int m = Core.getOptimalDFTSize(gray.rows());
            int n = Core.getOptimalDFTSize(gray.cols());
            Core.copyMakeBorder(gray, captusit, 0, m - gray.rows(), 0, n - gray.cols(), Core.BORDER_CONSTANT, Scalar.all(0));
            List<Mat> planes = new ArrayList<Mat>();
            captusit.convertTo(captusit, CvType.CV_32F);
            planes.add(captusit);
            planes.add(Mat.zeros(captusit.size(), CvType.CV_32F));
            Mat complexI = new Mat();
            Core.merge(planes, complexI);
            Core.dft(complexI, complexI);

            Core.split(complexI, planes);
            Core.magnitude(planes.get(0), planes.get(1), planes.get(0));
            Fourier = planes.get(0);

            Mat matOfOnes = Mat.ones(Fourier.size(), Fourier.type());
            Core.add(matOfOnes, Fourier, Fourier);
            Core.log(Fourier, Fourier);

            Fourier = Fourier.submat(new Rect(0, 0, Fourier.cols() & -2, Fourier.rows() & -2));
            int cx = Fourier.cols() / 2;
            int cy = Fourier.rows() / 2;
            Mat q0 = new Mat(Fourier, new Rect(0, 0, cx, cy));
            Mat q1 = new Mat(Fourier, new Rect(cx, 0, cx, cy));
            Mat q2 = new Mat(Fourier, new Rect(0, cy, cx, cy));
            Mat q3 = new Mat(Fourier, new Rect(cx, cy, cx, cy));
            Mat tmp = new Mat();
            q0.copyTo(tmp);
            q3.copyTo(q0);
            tmp.copyTo(q3);
            q1.copyTo(tmp);
            q2.copyTo(q1);
            tmp.copyTo(q2);
            Fourier.convertTo(Fourier, CvType.CV_8UC1);
            Core.normalize(Fourier, Fourier, 0, 255, Core.NORM_MINMAX, CvType.CV_8UC1);

            //Elementele care sunt folosite la eroziune si dilatare
            int kernelSize = 10;
            int elementType = Imgproc.CV_SHAPE_RECT;
            Mat element = Imgproc.getStructuringElement(elementType, new Size(2 * kernelSize + 1, 2 * kernelSize + 1), new Point(kernelSize, kernelSize));

            //erziune imagine
            Imgproc.erode(matrix, erode, element);

            //Dilatare imagine
            Imgproc.dilate(matrix, delate, element);


            //Evenimente adaugate butoanelor


            Original.setOnAction(new EventHandler<ActionEvent>() {
                @Override
                public void handle(ActionEvent e) {
                    imageView.setImage(image);
                }
            });


            Negative.setOnAction(new EventHandler<ActionEvent>() {
                @Override
                public void handle(ActionEvent e) {
                    imageView.setImage(mat2Image(negative));
                }
            });


            Grayscale.setOnAction(new EventHandler<ActionEvent>() {
                @Override
                public void handle(ActionEvent e) {
                    imageView.setImage(mat2Image(gray));
                }
            });


            Binary.setOnAction(new EventHandler<ActionEvent>() {
                @Override
                public void handle(ActionEvent e) {
                    imageView.setImage(mat2Image(albNegru));
                }
            });


            hsv.setOnAction(new EventHandler<ActionEvent>() {
                @Override
                public void handle(ActionEvent e) {
                    imageView.setImage(mat2Image(rgb));
                }
            });


            histograma.setOnAction(new EventHandler<ActionEvent>() {
                @Override
                public void handle(ActionEvent e) {
                    imageView.setImage(mat2Image(histImage));
                }
            });


            Mat finalFourier = Fourier;
            fourier.setOnAction(new EventHandler<ActionEvent>() {
                @Override
                public void handle(ActionEvent e) {
                    imageView.setImage(mat2Image(finalFourier));
                }
            });


            eroziune.setOnAction(new EventHandler<ActionEvent>() {
                @Override
                public void handle(ActionEvent e) {
                    imageView.setImage(mat2Image(erode));
                }
            });


            dilatare.setOnAction(new EventHandler<ActionEvent>() {
                @Override
                public void handle(ActionEvent e) {
                    imageView.setImage(mat2Image(delate));
                }
            });



        VBox box = new VBox();
        box.getChildren().addAll(Original, Negative, Grayscale, Binary, hsv, histograma, fourier, eroziune, dilatare);

        Border.setPrefWidth(1000);
        Border.setPrefHeight(600);


        Border.setCenter(imageView);
        Border.setRight(box);

        //Setarea pozitiei ImageView-ului
        imageView.setX(50);
        imageView.setY(25);
        imageView.setFitHeight(455);
        imageView.setFitWidth(500);
        imageView.setPreserveRatio(true);


        //Creem o scena
        Scene scene = new Scene(Border, 1000, 600);

        //Titlul scenei
        stage.setTitle("Image Processing");

        //adugam scena in platoul nostru
        stage.setScene(scene);

        //Afisam platoul
        stage.show();
    }
    public static void main(String args[]) {
        System.loadLibrary( Core.NATIVE_LIBRARY_NAME );
        launch(args);
    }
}