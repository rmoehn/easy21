(ns user
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.linear :as linear]
            [clojure.core.reducers :as r]
            [clojure.data :refer [diff]]
            [clojure.java.io :as io]
            [clojure.pprint :refer [pprint]]
            [clojure.repl :refer [pst doc find-doc]]
            [clojure.spec :as s]
            [clojure.spec.gen :as gen]
            [clojure.spec.test :as stest]
            [clojure.string :as string]
            [clojure.tools.namespace.repl :refer [refresh]]
            [easy21.core :as easy21 :refer :all]
            [incanter.core :as incanter]
            [incanter.charts :as charts]))

(m/set-current-implementation :vectorz)

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

        timestep-vectors
        (->> [(reset) 0 (init) nil]
             (iterate train-and-prep))

        ;some-timestep-vector
        ;(->> timestep-vectors
        ;     (drop 100)
        ;     first)

        ;experience (nth some-timestep-vector 2)

        ;the-v-matrx
        ;(-> experience
        ;    ::easy21/q
        ;    v-from-q
        ;    v-matrix)

        n 100000
        experiences (r/map #(get % 2) (r/take n timestep-vectors))

        _ (println "Finished training.")

        v-matrices
        (->> experiences
             (r/map ::easy21/q)
             (r/map v-from-q)
             (r/map v-matrix))

        ; Credits: https://math.stackexchange.com/questions/507742/distance-similarity-between-two-matrices
        differences
        (->> (partition 2 1 (into [] v-matrices))
             (r/map #(apply m/sub %))
             (r/map linear/norm)
             ;(r/partition 10)
             ;(r/map #(reduce + %))
             (into [])
             )

        ]
    (incanter/view (charts/line-chart (range (count differences)) differences))
    #_(spit "data.csv" (string/join \newline
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
