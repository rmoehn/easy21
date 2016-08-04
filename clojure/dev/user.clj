(ns user
  (:require [clojure.java.io :as io]
            [clojure.pprint :refer [pprint]]
            [clojure.repl :refer [pst doc find-doc]]
            [clojure.spec :as s]
            [clojure.spec.gen :as gen]
            [clojure.spec.test :as stest]
            [clojure.tools.namespace.repl :refer [refresh]]
            [easy21.core :as easy21 :refer :all]))

(comment

  (gen/generate (s/gen ::easy21/number))

  (reset)

  (s/conform (s/cat :new-state keyword?) [:bla])

  (stest/unstrument)
  (refresh)
  (stest/instrument (stest/instrumentable-syms))

  (pprint (until-done
            (iterate (stepper step dealer-think) [(reset) 0 nil nil])))


  )
