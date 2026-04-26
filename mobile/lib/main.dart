import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:file_selector/file_selector.dart';

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'Brain Tumor Detection',
        theme: ThemeData(
          colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        ),
        home: const HomePage(),
      );
}

class HomePage extends StatelessWidget {
  const HomePage({super.key});

  @override
  Widget build(BuildContext context) => Scaffold(
        appBar: AppBar(title: const Text('Brain Tumor Detection')),
        body: Center(
          child: ElevatedButton.icon(
            icon: const Icon(Icons.upload_file),
            label: const Text('Upload image or file'),
            onPressed: () => Navigator.of(context).push(
              MaterialPageRoute(builder: (_) => const UploadPage()),
            ),
          ),
        ),
      );
}

class UploadPage extends StatefulWidget {
  const UploadPage({super.key});

  @override
  State<UploadPage> createState() => _UploadPageState();
}

class _UploadPageState extends State<UploadPage> {
  String? _name;
  Uint8List? _bytes;
  bool _loading = false;

  Future<void> _pickFile() async {
    try {
      setState(() => _loading = true);
      final XFile? file = await openFile();
      if (file == null) return;
      final data = await file.readAsBytes();
      setState(() {
        _name = file.name;
        _bytes = data;
      });
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Could not pick file: $e')),
      );
    } finally {
      setState(() => _loading = false);
    }
  }

  Widget _preview() {
    if (_bytes == null) return const SizedBox.shrink();
    return Column(
      children: [
        Text(_name ?? '', style: const TextStyle(fontWeight: FontWeight.w600)),
        const SizedBox(height: 8),
        Image.memory(_bytes!, height: 240, fit: BoxFit.contain),
      ],
    );
  }

  @override
  Widget build(BuildContext context) => Scaffold(
        appBar: AppBar(title: const Text('Upload')),
        body: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              ElevatedButton.icon(
                onPressed: _loading ? null : _pickFile,
                icon: const Icon(Icons.attach_file),
                label: Text(_loading ? 'Picking...' : 'Choose file'),
              ),
              const SizedBox(height: 16),
              _preview(),
              const Spacer(),
              ElevatedButton(
                onPressed: _bytes == null
                    ? null
                    : () {
                        ScaffoldMessenger.of(context).showSnackBar(
                          const SnackBar(content: Text('Ready to upload')),
                        );
                      },
                child: const Text('Upload'),
              ),
            ],
          ),
        ),
      );
}
