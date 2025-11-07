# Contributing to PyTorch Android Mobile Application

First off, thank you for considering contributing to this project! 🎉

It's people like you that make this project such a great tool for the community.

## 🌟 Ways to Contribute

There are many ways you can contribute to this project:

- 🐛 **Report bugs** - Found a bug? Let us know!
- 💡 **Suggest features** - Have an idea? We'd love to hear it!
- 📝 **Improve documentation** - Help others understand the project better
- 🎨 **Improve UI/UX** - Make the app more beautiful and user-friendly
- 🔧 **Fix issues** - Pick an issue and submit a PR
- 🚀 **Add new features** - Bring new capabilities to the app
- 📖 **Write tutorials** - Help others learn from your experience

## 🚀 Getting Started

### Prerequisites

Before you begin, ensure you have:

- Android Studio Arctic Fox or newer
- JDK 11 or higher
- Git installed on your machine
- Basic knowledge of Android development and PyTorch

### Setting Up Your Development Environment

1. **Fork the repository**

   Click the "Fork" button at the top right of this page.

2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR-USERNAME/Pytorch-Android-Mobile-Application.git
   cd Pytorch-Android-Mobile-Application
   ```

3. **Add upstream remote**
   ```bash
   git remote add upstream https://github.com/umitkacar/Pytorch-Android-Mobile-Application.git
   ```

4. **Download the model**

   Download `model.pt` from [Google Drive](https://drive.google.com/file/d/1DG3dG4DKPnOQIfTE6RNqpvxp0dAD3CLQ/view?usp=sharing) and place it in:
   ```
   HelloWorldApp/app/src/main/assets/model.pt
   ```

5. **Open in Android Studio**

   Open the `HelloWorldApp` directory in Android Studio and let Gradle sync.

## 📋 Development Process

### 1. Create a Branch

Always create a new branch for your work:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

Use descriptive branch names:
- `feature/camera-capture` for new features
- `fix/memory-leak` for bug fixes
- `docs/api-documentation` for documentation
- `refactor/optimize-inference` for refactoring

### 2. Make Your Changes

- Write clean, readable code
- Follow the existing code style
- Add comments where necessary
- Update documentation if needed

### 3. Test Your Changes

Before submitting:

- ✅ Build the project successfully
- ✅ Test on physical device or emulator
- ✅ Verify existing features still work
- ✅ Test edge cases
- ✅ Check for memory leaks

### 4. Commit Your Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "feat: add camera capture functionality"
```

**Commit Message Format:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding tests
- `chore:` - Maintenance tasks

### 5. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 6. Create a Pull Request

1. Go to your fork on GitHub
2. Click "Compare & pull request"
3. Fill out the PR template
4. Link any related issues
5. Submit the PR

## 🎨 Code Style Guidelines

### Java Code Style

- Use **4 spaces** for indentation (not tabs)
- Follow [Google Java Style Guide](https://google.github.io/styleguide/javaguide.html)
- Use meaningful variable and method names
- Keep methods short and focused
- Add JavaDoc comments for public methods

**Example:**
```java
/**
 * Preprocesses the image for model inference.
 *
 * @param bitmap The input image bitmap
 * @return Preprocessed tensor ready for inference
 */
private Tensor preprocessImage(Bitmap bitmap) {
    Bitmap resized = Bitmap.createScaledBitmap(bitmap, 320, 320, false);
    return TensorImageUtils.bitmapToFloat32Tensor(
        resized,
        TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
        TensorImageUtils.TORCHVISION_NORM_STD_RGB
    );
}
```

### XML Layout Style

- Use meaningful IDs
- Follow Material Design guidelines
- Use ConstraintLayout for complex layouts
- Keep layouts simple and flat

## 🐛 Bug Reports

When filing a bug report, please include:

- **Description** - Clear description of the bug
- **Steps to reproduce** - Detailed steps to reproduce the issue
- **Expected behavior** - What you expected to happen
- **Actual behavior** - What actually happened
- **Screenshots** - If applicable
- **Environment**:
  - Device model
  - Android version
  - App version
- **Logs** - Relevant logcat output

## 💡 Feature Requests

When suggesting a feature:

- **Clear title** - Descriptive title for the feature
- **Problem statement** - What problem does it solve?
- **Proposed solution** - How should it work?
- **Alternatives** - Any alternative solutions considered?
- **Additional context** - Screenshots, mockups, examples

## 📝 Documentation

- Update README.md if you change functionality
- Add JavaDoc comments to new public methods
- Update inline comments if you change logic
- Create tutorials for complex features

## 🔍 Code Review Process

All submissions require review:

1. **Automated checks** - Must pass CI/CD checks
2. **Code review** - Maintainer reviews your code
3. **Testing** - Maintainer tests the changes
4. **Approval** - PR is approved and merged

**What reviewers look for:**
- Code quality and style
- Proper testing
- Documentation
- No breaking changes
- Performance implications

## 🏆 Recognition

Contributors will be:
- Listed in the README
- Mentioned in release notes
- Part of the project's success story!

## 📞 Getting Help

Need help? Here's how to reach out:

- 💬 **GitHub Discussions** - Ask questions and discuss ideas
- 🐛 **GitHub Issues** - Report bugs and request features
- 📧 **Email** - Contact the maintainer directly

## 📜 Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## ⚖️ License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Thank You!

Your contributions make this project better for everyone. We appreciate your time and effort!

---

**Happy Coding! 🚀**
