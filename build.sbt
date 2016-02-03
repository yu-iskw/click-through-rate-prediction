// Your sbt build file. Guides on how to write one can be found at
// http://www.scala-sbt.org/0.13/docs/index.html

scalaVersion := "2.10.5"

sparkVersion := "1.6.0"

crossScalaVersions := Seq("2.10.5", "2.11.7")

spName := "yu-iskw/click-through-rate-prediction"

// Don't forget to set the version
version := "0.0.1"

spAppendScalaVersion := true

spIncludeMaven := true

spIgnoreProvided := true

// Can't parallelly execute in test
parallelExecution in Test := false

fork in Test := true

javaOptions ++= Seq("-Xmx2G", "-XX:MaxPermSize=256m")

// All Spark Packages need a license
licenses := Seq("Apache-2.0" -> url("http://opensource.org/licenses/Apache-2.0"))

// Add Spark components this package depends on, e.g, "mllib", ....
sparkComponents ++= Seq("sql", "mllib")

libraryDependencies ++= Seq(
  "org.scalatest" %% "scalatest" % "2.1.5" % "test",
  "com.github.scopt" % "scopt_2.10" % "3.3.0"
)


// uncomment and change the value below to change the directory where your zip artifact will be created
// spDistDirectory := target.value

// add any Spark Package dependencies using spDependencies.
// e.g. spDependencies += "databricks/spark-avro:0.1"
spDependencies += "databricks/spark-csv:1.3.0-s_2.10"
