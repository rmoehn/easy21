(ns easy21.core
  (:require [clojure.spec :as s]
            [easy21.action :as action]))

(s/def ::color #{::color/black ::color/red})
(s/def ::number (s/int-in 1 11))
(s/def ::card (s/keys :req [::color ::number]))
(s/def ::hand (s/coll-of ::card :kind sequential?))
(s/def ::dealt-card ::card)
(s/def ::dealer-hand ::hand)
(s/def ::observation (s/keys :req [::dealt-card ::dealer-hand]))
(s/def ::action #{::action/hit ::action/stick})
(s/def ::player-hand ::hand)
(s/def ::env-state (s/keys :req [::player-hand ::dealer-hand]))
(s/def ::reward (s/int-in -1 2))
(s/def ::done? boolean?)

(s/fdef reset
  :args (s/cat)
  :ret (s/keys :req [::env-state ::observation]))

(s/fdef step
  :args (s/cat :env-state ::env-state :action ::action)
  :ret (s/keys :req [::env-state ::observation ::reward ::done?]))

(s/fdef init
  :args (s/cat :observation ::observation)
  :ret ::agent-state)

(s/fdef think
  :args (s/cat :agent-state ::agent-state :observation ::observation
               :reward ::reward)
  :ret (s/keys :req [::agent-state ::action]))

(defn play [initial-agent-state]
  (loop [agent-state initial-agent-state

         {:keys [::env-state ::observation]}
         (merge {::reward 0 ::done? false} (reset))]
    (let [{:keys [::new-agent-state ::action]}
          (think agent-state observation reward)

          {:keys [::env-state ::observation ::reward ::done?] :as env-response}
          (step env-state action)]
      (if-not done?
        (recur agent-state env-response)
        agent-state))))

(defn episode [[agent-state env-state] t]

  )

; DONE protocol:
;  1. Environment returns last reward and done=true.
;  2. Agent sees done=true, incorporates last reward, returns final experience
;     and action.
;  3. Control switches to just returning the previous values.
; OR:
;  1. If environment receives previous done=true state and action, it returns
;     previous state and reward 0.

(defn complete-step [{:keys [::state ::reward ::experience ::action]}]
  (if (::done? state)
    {::state state
     ::reward 0
     ::experience experience
     ::action nil})
  (let [{new-action ::action
         new-experience ::experience}
        (think experience (::observation state) reward)

        {new-state ::state
         new-reward ::reward}
        (step state action)]
    {::state new-state
     ::reward new-reward
     ::experience new-experience
     ::action new-action}))

(defn foo
  "I don't do a whole lot."
  [x]
  (println x "Hello, World!"))
