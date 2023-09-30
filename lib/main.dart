import 'dart:io';
import 'dart:math';

import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:collection/collection.dart';
import 'package:xrandom/xrandom.dart';

import 'package:flutter/material.dart';
import 'package:tiktoken/tiktoken.dart';

late final Interpreter interpreter;

class Message {
  late String message;
  late String sender;

  Message({required this.message, required this.sender});
  Map<String, dynamic> toJson() => {"message": message, "sender": sender};
  Message.fromJson(Map<String, dynamic> json) {
    message = json["message"];
    sender = json["sender"];
  }
}

void _copyAssetToLocal(sourceFile) async {
  try {
    var content = await rootBundle.load("assets/$sourceFile");
    final directory = await getApplicationDocumentsDirectory();
    var file = File("${directory.path}/$sourceFile");
    file.writeAsBytesSync(content.buffer.asUint8List());
  } catch (e) {}
}

var seqLen = 64;
var vocabSize = 50257;

padSequences(List a, int shape) {
  if (shape != a.length) {
    while (!(a.length == shape)) {
      a.insert(a.length, 0);
    }
  }
  return a;
}

int sample(List<dynamic> indexes, List<dynamic> probs) {
  var i = randomIndex(probs);
  return indexes[i];
}

int randomIndex(List<dynamic> probs) {
  var rnd = sumBy(probs) * Xrandom().nextFloat();
  var acc = 0.0;
  var i = 0;

  for (int j = 0; j < probs.length; j++) {
    i = j;
    acc += probs[j];
    if (rnd < acc) {
      break;
    }
  }

  return i;
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

void main() {
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  _MyApp createState() => _MyApp();
}

class _MyApp extends State<MyApp> {
  late List<Widget> _widgetOptions;
  late final Interpreter interpreter;
  late final TextEditingController controller;
  late final TextEditingController controllerN;
  late String result;
  late String punctuation;
  late Map tokenizer;
  late Map encoder;
  late Map responses;
  late Map jobs;
  late List messages;
  late final ScrollController _controller;
  late var tflite_model;
  late var data;
  int nbTokens = 100;

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

  @override
  void initState() {
    super.initState();
    controller = TextEditingController();
    controllerN = TextEditingController();
    result = "";
    _controller = ScrollController();
    punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""";
    messages = [];
    data = getData();
  }

  getData() async {
    final model_ = await _tflite_model;
    if (!await model_.exists()) {
      _copyAssetToLocal("model.tflite");
    }

    interpreter = Interpreter.fromFile(model_);
    return "";
  }

  Future<String> generateText(String prompt, Interpreter data) async {
    final encoding = encodingForModel("gpt2");
    var decodedToken = '';

    List<int> tokens = encoding.encode(prompt);

    for (int i = 0; i < nbTokens; i++) {
      var maxTokens =
          (tokens.length > seqLen) ? tokens.sublist(0, seqLen) : tokens;
      var paddedTokens = padSequences(maxTokens.toList(), seqLen);
      var inputIds = [paddedTokens];
      var predictions = [
        [
          for (var j = 1; j <= seqLen; j++)
            [for (var k = 1; k <= vocabSize; k++) 0.0]
        ]
      ];
      data.run(inputIds, predictions);
      var outputLogits = predictions[0][maxTokens.length - 1];
      var logits =
          outputLogits.mapIndexed((index, element) => {index: element});
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

      List logitsExp =
          filteredLogits.map((e) => exp(e - maxLogitValue)).toList();

      var sumExp = sumBy(logitsExp);

      var probs = logitsExp.map((e) => (e / sumExp)).toList();

      var logitsIndexes =
          filteredLogitsWithIndexes.map((e) => e.keys.toList()[0]).toList();

      var nextToken = sample(logitsIndexes, probs);

      tokens = tokens.toList();

      tokens.add(nextToken);

      decodedToken = encoding.decode(tokens);

      await Future.delayed(const Duration(microseconds: 1));

      setState(() {
        messages[messages.length - 1].message = decodedToken;
      });
    }
    return decodedToken;
  }

  Future<String> _getResponse(String message, Interpreter data) async {
    messages.add(Message(message: "", sender: "bot"));
    return await generateText(message, data);
  }

  int _selected = 0;
  void _onItemTapped(int index) {
    setState(() {
      _selected = index;
    });
  }

  updateNumberOfWords() {
    if (controllerN.text.isNotEmpty) {
      var input = controllerN.text;
      controllerN.text = "";
      nbTokens = int.parse(input);
    }
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
        title: 'Flutter Demo',
        theme: ThemeData(
          primarySwatch: Colors.blue,
        ),
        home: Builder(builder: (context) {
          _widgetOptions = <Widget>[
            FutureBuilder(
              builder: (context, snapshot) {
                if (snapshot.hasData) {
                  sent(a) {
                    if (controller.text.isNotEmpty) {
                      var input = controller.text;
                      controller.text = "";
                      messages.add(Message(message: input, sender: "you"));
                      setState(() {});
                      _getResponse(input, interpreter).then((value) {
                        messages.add(
                            Message(message: 'End of Message', sender: "bot"));
                        setState(() {});
                        Future.delayed(const Duration(milliseconds: 0), () {
                          _controller.animateTo(
                            _controller.position.maxScrollExtent,
                            curve: Curves.easeOut,
                            duration: const Duration(milliseconds: 500),
                          );
                        });
                        Future.delayed(const Duration(milliseconds: 200), () {
                          _controller.animateTo(
                            _controller.position.maxScrollExtent,
                            curve: Curves.easeOut,
                            duration: const Duration(milliseconds: 500),
                          );
                        });
                      });
                    }
                  }

                  if (snapshot.hasData) {
                    return Column(
                      children: [
                        Align(
                          alignment: Alignment.topLeft,
                          child: TextField(
                            controller: controllerN,
                            decoration:  InputDecoration(
                                border: const OutlineInputBorder(),
                                hintText: nbTokens.toString()),
                            keyboardType: TextInputType.number,
                            inputFormatters: <TextInputFormatter>[
                              FilteringTextInputFormatter.digitsOnly
                            ],
                            onSubmitted: updateNumberOfWords(),
                          ),
                        ),
                        Expanded(
                          child: ListView.builder(
                            controller: _controller,
                            itemBuilder: (context, index) {
                              List<InlineSpan> a = [];
                              for (var i
                                  in messages[index].message.split(" ")) {
                                try {
                                  if (Uri.parse(i).isAbsolute) {
                                    a.add(
                                      TextSpan(
                                        text: i + " ",
                                        style:
                                            const TextStyle(color: Colors.blue),
                                      ),
                                    );
                                  } else {
                                    a.add(TextSpan(
                                        text: i + " ",
                                        style: const TextStyle(
                                            color: Colors.black)));
                                  }
                                } on FormatException {
                                  a.add(TextSpan(
                                      text: i + " ",
                                      style: const TextStyle(
                                          color: Colors.black)));
                                }
                              }
                              return Container(
                                padding: const EdgeInsets.only(
                                    left: 16, right: 16, top: 10, bottom: 10),
                                child: Align(
                                  alignment: (messages[index].sender == "bot"
                                      ? Alignment.topLeft
                                      : Alignment.topRight),
                                  child: Container(
                                      decoration: BoxDecoration(
                                        borderRadius: BorderRadius.circular(15),
                                        color: (messages[index].sender == "bot"
                                            ? Colors.grey.shade200
                                            : Colors.blue[300]),
                                      ),
                                      padding: const EdgeInsets.all(16),
                                      child: RichText(
                                        text: TextSpan(children: a),
                                      )),
                                ),
                              );
                            },
                            itemCount: messages.length,
                            shrinkWrap: true,
                            padding: const EdgeInsets.only(top: 10, bottom: 10),
                          ),
                        ),
                        Align(
                          alignment: Alignment.bottomLeft,
                          child: Container(
                            padding: const EdgeInsets.only(
                                left: 10, bottom: 10, top: 10),
                            height: 60,
                            child: Row(
                              children: [
                                Flexible(
                                  child: TextField(
                                    controller: controller,
                                    decoration: const InputDecoration(
                                        border: OutlineInputBorder(),
                                        hintText: "Ask me something"),
                                    textInputAction: TextInputAction.go,
                                    onSubmitted: sent,
                                  ),
                                ),
                                IconButton(
                                  icon: const Icon(Icons.send),
                                  onPressed: () => sent(""),
                                ),
                              ],
                            ),
                          ),
                        )
                      ],
                    );
                  } else {
                    return const Center(child: CircularProgressIndicator());
                  }
                } else if (snapshot.hasError) {
                  print(snapshot.error);
                  return AlertDialog(
                    title: const Text("Restart"),
                    content: const Text("Please restart the app"),
                    actions: [
                      TextButton(
                          onPressed: () => SystemChannels.platform
                              .invokeMethod('SystemNavigator.pop'),
                          child: Text("Ok"))
                    ],
                  );
                } else {
                  return Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      crossAxisAlignment: CrossAxisAlignment.center,
                      children: const [
                        CircularProgressIndicator(),
                        Text("Downloading the required file")
                      ],
                    ),
                  );
                }
              },
              future: data,
            ),
            ListView(
              children: [
                ListTile(
                  title: const Text("Open source licence"),
                  onTap: () => showLicensePage(context: context),
                ),
              ],
            ),
          ];
          return Scaffold(
            resizeToAvoidBottomInset: true,
            bottomNavigationBar: BottomNavigationBar(
              items: const <BottomNavigationBarItem>[
                BottomNavigationBarItem(
                  icon: Icon(Icons.chat),
                  label: "Chat",
                ),
                BottomNavigationBarItem(
                  icon: Icon(Icons.assignment),
                  label: 'Log',
                ),
                BottomNavigationBarItem(
                  icon: Icon(Icons.settings),
                  label: 'Setting',
                ),
              ],
              currentIndex: _selected,
              selectedItemColor: Colors.amber[800],
              onTap: _onItemTapped,
            ),
            appBar: AppBar(title: const Text("Chatbot")),
            body: _widgetOptions.elementAt(_selected),
          );
        }));
  }
}
