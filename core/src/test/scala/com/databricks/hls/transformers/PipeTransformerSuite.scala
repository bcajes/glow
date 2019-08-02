package com.databricks.hls.transformers

import java.io.{BufferedInputStream, OutputStream}

import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.unsafe.types.UTF8String
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.catalyst.expressions.GenericInternalRow

import com.databricks.hls.sql.HLSBaseTest
import com.databricks.hls.transformers.pipe.{InputFormatter, InputFormatterFactory, OutputFormatter, OutputFormatterFactory, PipeTransformer}

class PipeTransformerSuite extends HLSBaseTest {
  test("read input and output formatters from service loader") {
    val sess = spark
    import sess.implicits._

    val df = Seq("dolphin").toDF.repartition(1)
    val options =
      Map("inputFormatter" -> "dummy_in", "outputFormatter" -> "dummy_out", "cmd" -> """["cat"]""")
    val output = new PipeTransformer().transform(df, options)
    assert(output.count() == 1)
    assert(output.schema.length == 1)
    assert(output.schema.exists(f => f.name == "animal" && f.dataType == StringType))
    assert(output.where("animal = 'monkey'").count() == 1)
  }
}

class DummyInputFormatterFactory() extends InputFormatterFactory {
  def name: String = "dummy_in"

  override def makeInputFormatter(df: DataFrame, options: Map[String, String]): InputFormatter = {
    new DummyInputFormatter()
  }
}

class DummyInputFormatter() extends InputFormatter {
  override def close(): Unit = ()

  override def write(record: InternalRow): Unit = ()

  override def init(stream: OutputStream): Unit = ()
}

class DummyOutputFormatterFactory() extends OutputFormatterFactory {
  override def name: String = "dummy_out"

  override def makeOutputFormatter(options: Map[String, String]): OutputFormatter = {
    new DummyOutputFormatter()
  }
}

class DummyOutputFormatter() extends OutputFormatter {
  override def makeIterator(stream: BufferedInputStream): Iterator[Any] = {
    val schema = StructType(Seq(StructField("animal", StringType)))
    val internalRow = new GenericInternalRow(
      Array(UTF8String.fromString("monkey")).asInstanceOf[Array[Any]])
    Iterator(schema, internalRow)
  }
}
