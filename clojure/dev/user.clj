(ns user
  (:require [clojure.core.matrix :as m]
            [clojure.data :refer [diff]]
            [clojure.java.io :as io]
            [clojure.pprint :refer [pprint]]
            [clojure.repl :refer [pst doc find-doc]]
            [clojure.spec :as s]
            [clojure.spec.gen :as gen]
            [clojure.spec.test :as stest]
            [clojure.string :as string]
            [clojure.tools.namespace.repl :refer [refresh]]
            [easy21.core :as easy21 :refer :all]))

(comment

  (gen/generate (s/gen ::easy21/number))

  (reset)

  (diff {:a 1 :b 1} {:a 1 :b 2})

  (s/conform (s/cat :new-state keyword?) [:bla])

  (refresh)

  (require '[easy21.core-test :as ct])
  (gen/sample ct/not-done-state-gen)

  (stest/check `step)

  (stest/unstrument)

  (refresh)

  (stest/instrument (stest/instrumentable-syms))
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
    (spit "data.csv" (string/join \newline
                                  (map #(string/join " " %)
                                       (m/emap double the-v-matrx)))))


  (require '[com.rpl.specter :refer [ALL END LAST] :as sr]
           '[com.rpl.specter.macros :as srm])

  (def experience {::episode [{::observation :bli0 ::action ::bla0 ::reward 4}
                              {::observation :bli1 ::action ::bla1}]})

  (srm/setval [::episode LAST ::reward] 5 experience)

  (srm/setval [::episode END] [{:b :bu}] experience)

  (require '[easy21.core-test :as ct] :reload)
  (stest/check `execute-e-policy
               {:get {::observation ct/inplay-observation-gen}})




  (gen/sample ct/inplay-observation-gen 5)


  )
