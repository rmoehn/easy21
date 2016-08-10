(ns easy21.VPlot
  (:import org.jzy3d.chart.factories.AWTChartComponentFactory
           [org.jzy3d.colors Color ColorMapper]
           org.jzy3d.colors.colormaps.ColorMapRainbow
           org.jzy3d.maths.Range
           [org.jzy3d.plot3d.builder Builder Mapper]
           org.jzy3d.plot3d.builder.concrete.OrthonormalGrid
           org.jzy3d.plot3d.primitives.Shape
           org.jzy3d.plot3d.rendering.canvas.Quality)
  (:gen-class
    :name easy21.VPlot
    :extends org.jzy3d.analysis.AbstractAnalysis
    :state _mapper
    :init new-instance
    :constructors {[org.jzy3d.plot3d.builder.Mapper] []}
    :exposes {chart {:get myGetChart :set setChart}}))

(defn -new-instance [mapper]
  [[] mapper])

(defn -init [this]
  (let [dealer-range (Range. -3 3)
        player-range (Range. -3 3)
        grid (OrthonormalGrid. dealer-range 80 player-range 80)
        ; 1 doesn't work. Strangely enough, it doesn't throw an
        ; ArithmeticException because of division by zero.
        surface (Builder/buildOrthonormal grid (._mapper this))
        color-mapper (ColorMapper. (ColorMapRainbow.)
                                   (-> surface .getBounds .getZmin)
                                   (-> surface .getBounds .getZmax)
                                   (Color. (float 1) (float 1) (float 1)
                                           (float 0.5)))
        the-chart (AWTChartComponentFactory/chart Quality/Advanced
                                                  (.getCanvasType this))]
    (doto surface
      (.setColorMapper color-mapper)
      (.setFaceDisplayed true)
      (.setWireframeDisplayed true))
    (-> the-chart .getScene .getGraph (.add surface))
    (.setChart this the-chart)))
