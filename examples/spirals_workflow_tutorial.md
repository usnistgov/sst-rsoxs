# Spirals workflow tutorial

The sample spot located from the bar image may have appeared good optically, but may not be the optimal spot for transmission measurements. To find a more optimal spot(s) and to gauge sample uniformity, quick RSoXS images can be captured on multiple spots in a spiral pattern on the same sample.  Follow the steps below to run spiral scans in Bluesky:


1. Run `waxs_spiral_mode()` in Bluesky.  This decreases the collection frequency of dark images, which otherwise are collected every time a change in the motor positions is detected. For spirals this eats a lot of time, and usually, this data is not used for rigorous analysis. However, if the intention is to use this data for rigorous analysis, skip this step. Note that anytime Bluesky is restarted, the default will be reset to `waxs_normal_mode()`, so `waxs_spiral_mode()` would need to be reactivated if continuing to run spiral scans.
2. Load a spreadsheet with a queue of spiral scans.  Typically, these are run at a single non-resonant energy (to minimize beam damage) and a single polarization (e.g., 0°).  Dry run the acquisition list and then start running the acquisitions.


## View spiral images in `nbs-viewer`

- With `nbs-viewer` open in the same environment as the Bluesky code on the beamline computer, click on the scan ID for the desired spiral scan in the left-hand table of scans.
- For the first spiral scan, right-click on this highlighted scan, select "Copy to New Display", and then select "Rsoxs-Spiral"
- For every subsequent spiral scan, right-click on this highlighted scan, select "Copy to Display", and then select "display_1"
- Navigate to the tab labelled "display_1"
- Under runs, select the desired spiral scan.  A grid of images should load in the right-hand panel, showing the spiral images organized by spatial location.
