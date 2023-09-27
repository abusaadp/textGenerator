import 'dart:io';
import 'dart:math';

import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:collection/collection.dart';
import 'package:xrandom/xrandom.dart';

import '../ui/home_page.dart';
import 'package:flutter/material.dart';
import 'package:tiktoken/tiktoken.dart';

late final Interpreter interpreter;

void _copyAssetToLocal(sourceFile) async {
  try {
    var content = await rootBundle.load("assets/$sourceFile");
    final directory = await getApplicationDocumentsDirectory();
    var file = File("${directory.path}/$sourceFile");
    file.writeAsBytesSync(content.buffer.asUint8List());
  } catch (e) {}
}

Future<String> get _localPath async {
  final directory = await getApplicationDocumentsDirectory();
  return directory.path;
}

Future<File> get _tflite_model async {
  final path = await _localPath;
  return File('$path/model.tflite');
}

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final model_ = await _tflite_model;
  if (!await model_.exists()) {
    _copyAssetToLocal("model.tflite");
  }
  interpreter = Interpreter.fromFile(model_);

  String text = 'Delhi is the capital of India';
  generate(text, interpreter);
  runApp(const SpAiSimpleTokenizer());
}

pad_sequences(List a, int shape) {
  if (shape != a.length) {
    while (!(a.length == shape)) {
      a.insert(a.length, 0);
    }
  }
  return a;
}

var seqLen = 64;
var vocab_size = 50257;

generate(text, Interpreter data) {
  final encoding = encodingForModel("gpt2");

  List<int> tokens = encoding.encode(text);

  for (int i = 0; i < 100; i++) {
    var maxTokens =
        (tokens.length > seqLen) ? tokens.sublist(0, seqLen) : tokens;
    var paddedTokens = pad_sequences(maxTokens.toList(), seqLen);
    var inputIds = [paddedTokens];
    var predictions = [
      [
        for (var j = 1; j <= seqLen; j++)
          [for (var k = 1; k <= vocab_size; k++) 0.0]
      ]
    ];
    data.run(inputIds, predictions);
    var outputLogits = predictions[0][maxTokens.length - 1];
    var logits = outputLogits.mapIndexed((index, element) => {index: element});
    Map<int, double> valueMap = new Map<int, double>();
    for (var element in logits) {
      valueMap.addEntries(
          {element.entries.first.key: element.entries.first.value}.entries);
    }
    var sortedByValueMap = Map.fromEntries(valueMap.entries.toList()
      ..sort((e2, e1) => e1.value.compareTo(e2.value)));

    var filteredLogitsWithIndexes = topK(sortedByValueMap, 40);

    var filteredLogits =
        filteredLogitsWithIndexes.map((e) => e.values.toList()[0]).toList();

    var maxLogitValue = max(filteredLogits);

    List logitsExp = filteredLogits.map((e) => exp(e - maxLogitValue)).toList();

    var sumExp = sumBy(logitsExp);

    var probs = logitsExp.map((e) => (e / sumExp)).toList();

    var logitsIndexes =
        filteredLogitsWithIndexes.map((e) => e.keys.toList()[0]).toList();

    var nextToken = sample(logitsIndexes, probs);

    tokens = tokens.toList();

    tokens.add(nextToken);

    var decodedToken = encoding.decode(tokens);

    print(decodedToken);
  }
}

int sample(List<dynamic> indexes, List<dynamic> probs) {
    var i = randomIndex(probs);
    return indexes[i];
}

int randomIndex(List<dynamic> probs) {
  var rnd = sumBy(probs) * Xrandom().nextFloat();
  var acc = 0.0;
  var i = 0;


  for(int j= 0 ; j < probs.length; j++) {
    i = j;
    acc += probs[j];
    if(rnd < acc) {
      break;
    }
  }

  return i ;
}

num sumBy(var numbers) {
  num sum = 0;
  for (var item in numbers) {
    sum += item;
  }
  return sum;
}

dynamic max(var logits) {
  var largestLogitValue = logits[0];

  for (var i = 0; i < logits.length; i++) {
    // Checking for largest value in the list
    if (logits[i] > largestLogitValue) {
      largestLogitValue = logits[i];
    }
  }
  return largestLogitValue;
}

/// Returns the [k] elements from [inputs].
List topK(Map<dynamic, dynamic> inputs, int k) {
  List q = [];

  inputs.forEach((key, value) {
    q.add({key: value});
    if (q.length > k) {
      q.removeLast();
    }
  });
  return q.toList();
}

toDoubleArray(array) {
  var arr = List.filled(array.length, 0.0);
  for (int i = 0; i < array.length; i++) {
    arr[i] = array[i];
  }
  return arr;
}

class SpAiSimpleTokenizer extends StatelessWidget {
  const SpAiSimpleTokenizer({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'A Simple BPE Tokenizer Demo',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
          colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
          useMaterial3: true),
      home: const HomePage(),
    );
  }
}
