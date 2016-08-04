(ns easy21.core
  (:require [clojure.spec :as s]
            [clojure.spec.gen :as gen]
            [easy21.action :as action]
            [easy21.color :as color]))

(s/def ::black-card (s/int-in 1 11))
(s/def ::red-card (s/int-in -1 -11))
(s/def ::card (s/or :black ::black-card :red ::red-card))
(s/def ::cards (s/coll-of ::card :kind sequential?))
(s/def ::sum (s/int-in -9 32))
(s/def ::player-sum ::sum)
(s/def ::dealer-sum ::sum)
(s/def ::observation (s/keys :req [::player-sum ::dealer-sum]))
(s/def ::action #{::action/hit ::action/stick})
(s/def ::state (s/keys :req [::observation ::done?]))
(s/def ::reward (s/int-in -1 2))
(s/def ::done? boolean?)

(s/def ::experience true)

(s/fdef reset
  :args (s/cat)
  :ret ::state)

(declare bust?)

(defn stick-and-done? [{{:keys [action]} :args
                       {:keys [new-state]} :ret}]
  (and (= ::action/stick action) (::done? new-state)))

(defn bust-and-done? [{{{:keys [observation] :as new-state} :new-state} :ret}]
  (and (or (bust? (::player-sum observation))
           (bust? (::dealer-sum observation)))
       (::done? new-state)))

(defn single-bust? [{{{:keys [observation]} :new-state} :ret}]
  (not (and (bust? (::player-sum observation))
            (bust? (::dealer-sum observation)))))

(s/fdef step
  :args (s/cat :state ::state :action ::action)
  :ret (s/cat :new-state ::state :new-reward ::reward)
  :fn (s/and stick-and-done? bust-and-done? single-bust?))

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
    (let [[new-experience new-action]
          (think experience (::observation state) reward)

          [new-state new-reward]
          (step state new-action)]
      [new-state new-reward new-experience new-action])))

(defn until-done [timesteps]
  (let [[not-done done] (split-with #(not (::done? (first %))) timesteps)]
    (conj (vec not-done) (first done))))

(defn rand-number []
  (inc (rand-int 10)))

(defn rand-color []
  (if (zero? (rand-int 3))
    -1
    1))

(defn rand-card []
  (* (rand-color) (rand-number)))

(defn reset []
  {::observation {::player-sum (rand-number)
                  ::dealer-sum (rand-number)}
   ::done? false})

(defn bust? [player-sum]
  (not (s/int-in-range? 1 22 player-sum)))

(defmulti step (fn [_ action] action))

(defmethod step ::action/hit [{prev-obs ::observation :as prev-state} _]
  (let [new-player-sum (+ (::player-sum prev-obs) (rand-card))]
    [(-> prev-state
         (assoc-in [::observation ::player-sum] new-player-sum)
         (assoc ::done? (bust? new-player-sum)))
     (if (bust? new-player-sum) -1 0)]))

;;; Note: I wish I could implement the dealer as a deterministic strategy with
;;; the same signature as `think`. However, I would have to extend `step`, so
;;; that it can react to both player and dealer actions. Also, I'd have to
;;; introduce recursion. This is all complicated, so I leave it at the threading
;;; form below.
(defmethod step ::action/stick [{prev-obs ::observation :as prev-state} _]
  (let [final-dealer-sum
        (->> (::dealer-sum prev-obs)
             (iterate #(+ % (rand-card)))
             (drop-while #(not (or (bust? %) (>= % 17))))
             first)

        reward
        (if (bust? final-dealer-sum)
          1
          (Integer/signum (- (::player-sum prev-obs) final-dealer-sum)))]
    [{::observation {::player-sum (::player-sum prev-obs)
                     ::dealer-sum final-dealer-sum}
      ::done? true}
     reward]))

(defn dealer-think [_ observation _]
  (if (>= (::player-sum observation) 17)
    [nil ::action/stick]
    [nil ::action/hit]))

(comment
  (iterate (stepper step think) [(init) 0 current-experience nil]))
