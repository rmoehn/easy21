(ns user
  (:require [clojure.java.io :as io]
            [clojure.pprint :refer [pprint]]
            [clojure.repl :refer [pst doc find-doc]]
            [clojure.spec :as s]
            [clojure.spec.gen :as gen]
            [clojure.spec.test :as stest]
            [clojure.tools.namespace.repl :refer [refresh]]
            [easy21.core :as easy21 :refer :all])
  (:import [org.jzy3d.chart Chart ChartLauncher]
           org.jzy3d.plot3d.rendering.canvas.Quality
           [org.jzy3d.colors Color ColorMapper]
           org.jzy3d.colors.colormaps.ColorMapRainbow
           org.jzy3d.maths.Range
           [org.jzy3d.plot3d.builder Builder Mapper]
           org.jzy3d.plot3d.builder.concrete.OrthonormalGrid
           org.jzy3d.plot3d.rendering.canvas.Quality))

(defn sin-mapper []
  (proxy [Mapper] []
   (f ([double-ary-ary] (proxy-super f double-ary-ary))
      ([x y]
       (if (= (type x) double-ary-type)
          (proxy-super f x y)
          (* x (Math/sin (* x y))))))))

(defn show-thing [mapper]
  (let [dealer-range (Range. -3 3)
        player-range (Range. -3 3)
        grid (OrthonormalGrid. dealer-range 80 player-range 80)
        ; 1 doesn't work. Strangely enough, it doesn't throw an
        ; ArithmeticException because of division by zero.
        surface (Builder/buildOrthonormal grid mapper)
        color-mapper (ColorMapper. (ColorMapRainbow.)
                                   (-> surface .getBounds .getZmin)
                                   (-> surface .getBounds .getZmax)
                                   (Color. (float 1) (float 1) (float 1)
                                           (float 0.5)))
        chart (Chart. Quality/Advanced)]
    (doto surface
      (.setColorMapper color-mapper)
      (.setFaceDisplayed true)
      (.setWireframeDisplayed true))
    (-> chart .getScene .getGraph (.add surface))
    (ChartLauncher/openChart chart)))

(comment

  (show-thing (sin-mapper))


  (gen/generate (s/gen ::easy21/number))

  (reset)

  (s/conform (s/cat :new-state keyword?) [:bla])

  (refresh)

  (require '[easy21.core-test :as ct])
  (gen/sample ct/not-done-state-gen)

  (stest/check `step)

  (stest/unstrument)

  (compile 'easy21.VPlot)

  (refresh)
  (import 'easy21.VPlot)

  (AnalysisLauncher/open (VPlot. (sin-mapper)))

  ;(stest/instrument (stest/instrumentable-syms))
  (let [complete-step (stepper step policy-think)
        train-and-prep (make-train-and-prep reset init complete-step wrapup)

        some-timestep-vector
        (->> [(reset) 0 (init) nil]
             (iterate train-and-prep)
             (drop 100)
             first)

        experience (nth some-timestep-vector 2)

        the-v-matrx
        (-> experience
            ::easy21/q
            v-from-q
            v-matrix)]
    (AnalysisLauncher/open (VPlot. (make-mapper the-v-matrx))))


  (require '[com.rpl.specter :refer [ALL END LAST] :as sr]
           '[com.rpl.specter.macros :as srm])

  (def experience {::episode [{::observation :bli0 ::action ::bla0 ::reward 4}
                              {::observation :bli1 ::action ::bla1}]})

  (srm/setval [::episode LAST ::reward] 5 experience)

  (srm/setval [::episode END] [{:b :bu}] experience)

  (srm/transform
    #()
    {:a {:b 4}
     :c {:d 5}})

  (defn make-point []
    (proxy [java.awt.Point] []
      (setLocation [the-x the-y]
        (set! (. this x) the-x)
        (set! (. this y) the-y))))

  (def point (make-point))

  (.setLocation point 4 5)

  (str ^java.awt.Point point)

  cast

  )
