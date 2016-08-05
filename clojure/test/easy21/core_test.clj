(ns easy21.core-test
  (:require [clojure.spec :as s]
            [clojure.spec.gen :as gen]
            [clojure.test :refer :all]
            [easy21.core :as nut]))

(def not-done-state-gen
  {::nut/done? false
   ::nut/observation (gen/hash-map ::nut/dealer-sum (s/gen ::nut/black-card)
                                   ::nut/player-sum (s/gen (s/int-in 1 22)))})

(deftest a-test
  (testing "FIXME, I fail."
    (is (= 0 1))))
