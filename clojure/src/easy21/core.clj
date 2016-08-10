(ns easy21.core
  (:require [clojure.core.matrix :as m]
            [clojure.spec :as s]
            [clojure.spec.gen :as gen]
            [com.rpl.specter :refer [ALL END LAST] :as sr]
            [com.rpl.specter.macros :as srm]
            [easy21.action :as action]
            [easy21.color :as color])
  (:import org.jzy3d.plot3d.builder.Mapper))

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

(s/def ::policy (s/map-of ::observation ::action))
(s/def ::n0 integer?)
(s/def ::nseen-s (s/map-of ::observation nat-int?))
(s/def ::nseen-sa (s/map-of ::observation (s/map-of ::action nat-int?)))
(s/def ::q (s/map-of ::observation (s/map-of ::action number?)))
(s/def ::timestep (s/or :butlast (s/keys :req [::observation ::reward ::action])
                        :last (s/keys :req [::observation ::reward])))
(s/def ::episode (s/coll-of ::timestep :kind sequential?))
(s/def ::experience (s/keys :req [::policy ::n0 ::nseen-s ::nseen-sa
                                  ::episode ::q]))

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

(s/fdef wrapup
  :args (s/cat :experience ::experience :observation ::observation
               :reward ::reward)
  :ret ::experience)

(declare wrapup)

(s/def ::timestep-vec (s/cat :state ::state
                             :reward ::reward
                             :experience ::experience
                             :action ::action))

(s/fdef stepper
  :ret (s/fspec :args (s/cat :arg ::timestep-vec)
                :ret ::timestep-vec))

;; The first timestep has an observation and the action decided based on that
;; observation, but the reward is 0. The algorithms don't look at that reward.
;; The last timestep has an observation and a reward, but no action. The return
;; for the last timestep is undefined.
(defn stepper [step think]
  (fn complete-step [[state reward experience action]]
    (if (::done? state)
      [state 0 experience nil]
      (let [[new-experience new-action]
            (think experience (::observation state) reward)

            [new-state new-reward]
            (step state new-action)]
        [new-state new-reward new-experience new-action]))))

(s/def ::reset-state (s/and ::state
                            #(not (::done? %))
                            #(s/valid? ::black-card (::player-sum %))
                            #(s/valid? ::black-card (::dealer-sum %))))
(s/def ::fresh-experience (s/and ::experience
                                 #(empty? (::episode %))))
(s/def ::first-timestep-vec (s/cat :state ::reset-state
                                   :reward zero?
                                   :experience ::fresh-experience
                                   :action nil?))

(defn until-done [timesteps]
  (let [[not-done done] (split-with #(not (::done? (first %))) timesteps)]
    (conj (vec not-done) (first done))))

(s/fdef make-train-and-prep
  :ret (s/fspec
         :args (s/cat :arg ::first-timestep-vec)
         :ret ::first-timestep-vec))

(defn make-train-and-prep [reset init complete-step wrapup]
  (fn train-and-prep [first-timestep-vec]
    (let [episode
          (->> first-timestep-vec
               (iterate complete-step)
               until-done)

          [state reward experience _] (last episode)

          experience (wrapup experience (::observation state) reward)]
      [(reset) 0 experience nil])))

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
   ::nseen-s {}
   ::nseen-sa {}
   ::episode []
   ::q {}})

(defn rand-action []
  (rand-nth [::action/stick ::action/hit]))

(defn rand-explore? [n0 nseen-s]
  (< (rand-int (+ n0 nseen-s)) n0)) ; true with probability n0 / (n0 + nseen-s)

(defn execute-e-policy [{:keys [::policy ::n0 ::nseen-s] :as experience}
                        observation]
  (if (zero? (get nseen-s observation 0))
    (let [action (rand-action)]
      [(-> experience
           (assoc-in [::policy observation] action)
           (assoc-in [::nseen-s observation] 1))
       action])
    [(update-in experience [::nseen-s observation] inc)
     (if (rand-explore? n0 (get nseen-s observation))
       (rand-action)
       (get-in experience [::policy observation]))]))

(s/fdef policy-think
  :args (s/cat :experience ::experience :observation ::observation
               :reward ::reward)
  :ret (s/cat ::experience ::action))

(defn policy-think [experience observation reward]
  (let [[experience action] (execute-e-policy experience observation)]
    [(srm/setval [::episode END] [{::observation observation
                                   ::reward reward
                                   ::action action}]
                 experience)
     action]))

(defn wrapup [{:keys [::policy ::episode ::nseen-sa ::q] :as experience}
              observation reward]
  (let [episode
        (srm/setval [END] [{::observation observation ::reward reward}]
                    episode)

        g-t (->> episode
                 (map ::reward)
                 reverse
                 (reductions +)
                 reverse
                 rest)
        timesteps-with-g-t (map #(assoc %1 ::g-t %2) episode g-t)

        obs-action-freqs
        (->> episode
             butlast
             (map #(dissoc % ::reward))
             frequencies)

        new-nseen-sa
        (reduce-kv (fn [nseen-sa {:keys [::observation ::action]} nseen]
                     (update-in nseen-sa [observation action]
                                #(+ (or % 0) nseen)))
                   nseen-sa
                   obs-action-freqs)

        new-q
        (reduce (fn [q {:keys [::observation ::action ::g-t]}]
                  (update-in q [observation action]
                             #(let [old-q (or % 0)]
                                (+ old-q (* (/ 1 (get-in new-nseen-sa
                                                         [observation action]))
                                            (- g-t old-q))))))
                q
                timesteps-with-g-t)

        ; Credits: https://clojuredocs.org/clojure.core/max-key#example-5490032de4b09260f767ca79
        new-policy
        (into {} (map #(vector % (first (apply max-key val (get new-q %))))
                      (keys policy)))]
    (-> experience
        (assoc ::episode [])
        (assoc ::nseen-sa new-nseen-sa)
        (assoc ::policy new-policy)
        (assoc ::q new-q))))

(defn dealer-think [_ observation _]
  (if (>= (::player-sum observation) 17)
    [nil ::action/stick]
    [nil ::action/hit]))

(defn v-from-q [q]
  (into {} (map #(vector (key %)
                         (apply max (vals (val %))))
                q)))

(defn v-matrix [v]
  (let [m (m/new-matrix 10 21)]
    (reduce-kv (fn [matrix {:keys [::dealer-sum ::player-sum]} value]
              (m/mset matrix (dec dealer-sum) (dec player-sum) value))
            m
            v)))

;; Credits: http://stackoverflow.com/a/9090720/5091738
(def double-ary-type (Class/forName "[D"))

(defn make-mapper [v-matrix]
  (proxy [Mapper] []
    (f ([double-ary-ary] (proxy-super f double-ary-ary))
      ([dealer-card player-sum]
       (if (= (type dealer-card) double-ary-type)
          (proxy-super f dealer-card player-sum)
          (m/mget v-matrix (dec dealer-card) (dec player-sum)))))))

(comment
  (iterate (stepper step think) [(init) 0 current-experience nil]))
