(ns easy21.core
  (:require [clojure.spec :as s]
            [clojure.spec.gen :as gen]
            [easy21.action :as action]
            [easy21.color :as color]))

(declare bust?)

(defn implies [a b]
  (or (not a) b))

(defn bust-then-done? [{:keys [::observation] :as state}]
  (implies (or (bust? (::player-sum observation))
               (bust? (::dealer-sum observation)))
           (::done? state)))

(defn single-bust? [observation]
  (not (and (bust? (::player-sum observation))
            (bust? (::dealer-sum observation)))))

(defn dealer-hit-then-done? [{:keys [::observation] :as state}]
  (implies (not (s/valid? ::black-card (::dealer-sum observation)))
           (::done? state)))

(s/def ::black-card (s/int-in 1 11))
(s/def ::red-card (s/int-in -1 -11))
(s/def ::card (s/or :black ::black-card :red ::red-card))
(s/def ::cards (s/coll-of ::card :kind sequential?))
(s/def ::sum (s/int-in -9 32))
(s/def ::player-sum ::sum)
(s/def ::dealer-sum ::sum)
(s/def ::observation (s/and (s/keys :req [::player-sum ::dealer-sum])
                            single-bust?))
(s/def ::action #{::action/hit ::action/stick})
(s/def ::state (s/and (s/keys :req [::observation ::done?])
                      bust-then-done?
                      dealer-hit-then-done?))
(s/def ::reward (s/int-in -1 2))
(s/def ::done? boolean?)

(s/def ::experience true)

(s/fdef reset
  :args (s/cat)
  :ret ::state)

(defn stick-and-done? [{{:keys [action]} :args
                       {:keys [new-state]} :ret}]
  (implies (= ::action/stick action) (::done? new-state)))

(s/fdef step
  :args (s/cat :state (s/and ::state #(not (::done? %))) :action ::action)
  :ret (s/cat :new-state ::state :new-reward ::reward)
  :fn (s/and stick-and-done?))

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

(defn init []
  {::policy {}
   ::n0 100
   ::nseen {}
   ::episode []})

(defn rand-action []
  (rand-nth [::action/stick ::action/hit]))

(defn rand-explore? [n0 nseen]
  (< (rand-int (+ n0 nseen) n0))) ; true with probability n0 / (n0 + nseen)

(defn policy [{:keys [::policy ::n0 ::nseen] :as experience} observation]
  (if (zero? (get nseen observation 0))
    (let [action (rand-action)]
      [(-> experience
           (assoc-in [::policy observation] action)
           (assoc-in [::nseen observation] 1))
       action])
    (if (rand-explore? n0 (get nseen observation))
      [experience (rand-action)]
      [(update-in experience [::nseen observation] inc)
       (get-in experience [::policy observation])])))

(defn policy-think [experience observation reward]
  (let [[experience action] (policy experience observation)]
    [(update experience ::episode
             #(conj % {::observation observation
                       ::reward reward
                       ::action action}))
     action]))

(defn wrapup [{:keys [::episode ::n-s-a ::q]}]
  (let [g-t (->> episode
                 (map ::reward)
                 reverse
                 (reductions +)
                 reverse)
        timesteps-with-g-t (map #(assoc %1 ::g-t %2) episode g-t)

        new-q
        (reduce (fn [{:keys [::observation ::action ::g-t]} q]
                  (update-in q [observation timestep]
                             #(let [old-q (or % 0)]
                                (+ old-q (* (/ 1 (get-in n-s-a
                                                         [observation action]))
                                            (- g-t old-q))))))
                q
                timesteps-with-g-t)

        ; Credits: https://clojuredocs.org/clojure.core/max-key#example-5490032de4b09260f767ca79
        new-policy
        (into {} (map #(vec % (apply max-key val (get q %)))
                      policy))]))

(defn dealer-think [_ observation _]
  (if (>= (::player-sum observation) 17)
    [nil ::action/stick]
    [nil ::action/hit]))

(comment
  (iterate (stepper step think) [(init) 0 current-experience nil]))
