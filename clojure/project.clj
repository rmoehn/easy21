(defproject easy21 "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.9.0-alpha10"]
                 [org.clojure/test.check "0.9.0"]
                 [com.rpl/specter "0.12.0"]
                 [net.mikera/core.matrix "0.52.2"]
                 [org.jzy3d/jzy3d-api "0.9.1"]]
  :repositories [["jzy3d-releases" "http://maven.jzy3d.org/releases"]]

  :profiles {:dev {:dependencies [[org.clojure/tools.namespace "0.2.11"]]
                   :source-paths ["dev"]}})
