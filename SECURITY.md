# Security Policy

## 🔒 Supported Versions

We actively support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| Latest  | ✅ Yes             |
| < 1.0   | ❌ No              |

## 🐛 Reporting a Vulnerability

We take the security of PyTorch Android Mobile Application seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### 📧 How to Report

**Please DO NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via:

1. **Email**: Contact the repository owner directly through GitHub
2. **Private Security Advisory**: Use GitHub's private vulnerability reporting feature

### 📋 What to Include

Please include the following information:

- **Type of issue** (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- **Full paths** of source file(s) related to the issue
- **Location** of the affected source code (tag/branch/commit or direct URL)
- **Step-by-step instructions** to reproduce the issue
- **Proof-of-concept or exploit code** (if possible)
- **Impact** of the issue, including how an attacker might exploit it

### ⏱️ Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 5 business days
- **Status Update**: Every 7 days until resolved
- **Fix Timeline**: Varies based on severity
  - Critical: 7 days
  - High: 30 days
  - Medium: 90 days
  - Low: Best effort

### 🎯 What to Expect

1. **Acknowledgment** - We'll confirm receipt of your report
2. **Investigation** - We'll investigate and validate the issue
3. **Resolution** - We'll work on a fix
4. **Disclosure** - We'll coordinate disclosure with you
5. **Credit** - We'll publicly thank you (if you wish)

## 🛡️ Security Best Practices

When using this application:

### For Users

- ✅ Only download models from trusted sources
- ✅ Keep the app updated to the latest version
- ✅ Review app permissions before granting
- ✅ Don't process sensitive or personal images if privacy is a concern
- ✅ Be aware that on-device processing means your images stay on your device

### For Developers

- ✅ Validate all inputs from the model
- ✅ Use secure model loading practices
- ✅ Implement proper error handling
- ✅ Keep dependencies up to date
- ✅ Follow Android security best practices
- ✅ Use ProGuard/R8 for release builds
- ✅ Implement certificate pinning if networking is added

## 🔐 Known Security Considerations

### Model Security

- **Model Integrity**: Always verify the source of PyTorch models
- **Model Tampering**: Models should be downloaded from trusted sources only
- **Adversarial Attacks**: Be aware that ML models can be fooled by adversarial examples

### App Permissions

This app requires:
- `READ_EXTERNAL_STORAGE` - To read images from gallery
- No network permissions (fully offline)

### Data Privacy

- All image processing happens **on-device**
- No data is sent to external servers
- Images are not stored or logged by the app

## 📚 Security Resources

- [Android Security Best Practices](https://developer.android.com/topic/security/best-practices)
- [PyTorch Security](https://pytorch.org/docs/stable/community/contribution_guide.html#security)
- [OWASP Mobile Security](https://owasp.org/www-project-mobile-security/)

## 🏆 Security Hall of Fame

We appreciate security researchers who help us keep this project secure:

<!-- Contributors who report security issues will be listed here -->

*No vulnerabilities reported yet - be the first!*

## 📄 Disclosure Policy

- We follow **coordinated disclosure**
- We aim to patch vulnerabilities before public disclosure
- We'll credit security researchers (with permission)
- We'll publish security advisories for significant issues

## ⚖️ Safe Harbor

We support safe harbor for security researchers who:

- Make a good faith effort to avoid privacy violations
- Only interact with accounts you own or with explicit permission
- Do not exploit vulnerabilities beyond demonstration
- Report vulnerabilities promptly
- Keep vulnerabilities confidential until they are resolved

---

**Thank you for helping keep PyTorch Android Mobile Application and its users safe! 🙏**
