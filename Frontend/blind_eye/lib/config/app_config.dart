/// Walking Eye - App Configuration
///
/// FOR LOCAL TESTING (recommended during the hackathon — no internet
/// dependency, fastest to iterate on):
///   1. Run the backend locally (uvicorn ...).
///   2. Find your laptop's local IP: run `ipconfig` in a Windows
///      terminal and look for "IPv4 Address" under your WiFi adapter
///      (e.g. 192.168.1.42).
///   3. Set baseUrl below to 'http://<that IP>:8000' — NOT 'localhost',
///      since your phone can't resolve your laptop's localhost.
///   4. Make sure your phone and laptop are on the same WiFi network.
///
/// FOR THE DEPLOYED BACKEND (e.g. showing the app without your laptop
/// nearby, or a backup if local WiFi is unreliable on demo day):
///   Set baseUrl to your Render URL instead.
class AppConfig {
  static const String baseUrl =
      'http://192.168.1.88:8000'; // <-- change this to your laptop's IP
  static const String analyzeEndpoint = '$baseUrl/analyze';
  static const String healthEndpoint = '$baseUrl/health';

  /// Timeout for analysis requests in seconds
  static const int requestTimeoutSeconds = 30;

  /// How often to send a frame to the backend (milliseconds)
  /// 1500ms = safe for CPU inference + isolate conversion
  static const int streamIntervalMs = 1500;
}
