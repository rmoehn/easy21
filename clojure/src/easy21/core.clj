(ns easy21.core
  (:require [clojure.spec :as s]
            [easy21.action :as action]
            [easy21.color :as color]))

(s/def ::color #{::color/black ::color/red})
(s/def ::number (s/int-in 1 11))
(s/def ::card (s/keys :req [::color ::number]))
(s/def ::hand (s/coll-of ::card :kind sequential?))
(s/def ::player-sum (s/int-in -9 32))
(s/def ::dealer-hand ::hand)
(s/def ::observation (s/keys :req [::player-sum ::dealer-hand]))
(s/def ::action #{::action/hit ::action/stick})
(s/def ::state (s/keys :req [::observation ::done?]))
(s/def ::reward (s/int-in -1 2))
(s/def ::done? boolean?)

(s/def ::experience some?)

(s/fdef reset
  :args (s/cat)
  :ret ::state)

(s/fdef step
  :args (s/cat :state ::state :action ::action)
  :ret (s/cat :new-state ::state :new-reward ::reward))

(s/fdef init
  :args (s/cat)
  :ret ::experience)

(s/fdef think
  :args (s/cat :experience ::experience :observation ::observation
               :reward ::reward)
  :ret (s/cat :new-experience ::experience :action ::action))

(defn stepper [step think]
  (fn complete-step [[state reward experience action]]
    (if (::done? state)
      [state 0 experience nil])
    (let [[new-action new-experience]
          (think experience (::observation state) reward)

          [new-state new-reward]
          (step state action)]
      [new-state new-reward new-experience new-action])))

(comment
  (iterate (stepper step think) [(init) 0 current-experience nil]))
