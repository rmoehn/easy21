(ns easy21.two-round-nd
  "A Clojure implementation of OpenAI Gym's two round nondeterministic reward
  environment."
  (:require [easy21.core :as e]))

(defn reset []
  {::e/observation 2
   ::e/done? false})

(def reward-for [[[-1 1] [0 0 9]]
                 [[0 2]  [2 3]]])

; Note: Observation 0, 1 → action taken before; 2 → starting state.
(defn step [{:keys [::e/observation]} action]
  [{::e/observation action
    ::e/done? (not= observation 2)}
   (rand-nth (get-in reward-for [observation action] [0]))])
      ; In the second step, the observation indicates the previous action and
      ; therefore the reward is determined by the first and second action.

(defn rand-action []
  (rand-int 2))

(defn v-vector [v-map]
  (mapv #(get v-map % 0) [0 1 2]))
