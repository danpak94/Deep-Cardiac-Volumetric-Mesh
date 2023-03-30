import importlib
import qt
import slicer

import pyvista as pv

class CustomQtSignalSender(qt.QObject):
    # https://discourse.slicer.org/t/how-to-use-signals-and-slots-in-slicer-3d/14013/5
    # https://discourse.slicer.org/t/use-of-qt-signal-leads-to-a-crash-on-exit/8321
    signal = qt.Signal(object) # we have to put this line here (not inside __init__) to prevent crashing.. idk why but it's important...
    def __init__(self):
        super(CustomQtSignalSender, self).__init__(None)

class ProgressBarAndRunTime():
    def __init__(self, progressBar):
        self.progressBar = progressBar
        self.timer = qt.QElapsedTimer()

    def start(self, value=0, maximum=1, text='Running ... %v / {}'):
        self.progressBar.value = value
        self.progressBar.maximum = maximum
        self.progressBar.setFormat(text.format(maximum))
        self.progressBar.show()
        slicer.app.processEvents() # https://github.com/Slicer/Slicer/blob/a5f75351073ef62fd6198d9480d86c0009d70f9b/Modules/Scripted/DICOMLib/DICOMSendDialog.py
        self.timer.start()

    def step(self, value):
        if value < self.progressBar.maximum:
            self.progressBar.value = value
            self.progressBar.setFormat('Running ... %v / {}'.format(self.progressBar.maximum))
        else:
            self.end()

    def end(self, text='Done: {} seconds'):
        time_elapsed = self.timer.elapsed() / 1000 # originally in milliseconds
        self.progressBar.setValue(self.progressBar.maximum)
        self.progressBar.setFormat(text.format(time_elapsed))