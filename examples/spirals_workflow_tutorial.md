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
- If needed, double-click on an image to view the indivdual image zoomed in.


## Guidance for deciding optimal location(s) from spiral scan

It is strongly recommended to pick at least 3 spots per sample to ensure repeatability and homogeneity across the sample.  For assistance in identifying what "good" spots look like, see the Guide to Picking Spiral Scans: https://wiki-nsls2.bnl.gov/beamline7ID1/index.php?title=Guide_to_Picking_Spiral_Scans.  More tips are included below.
- In the image itself, check that there is sufficient flux (e.g., the beam is not blocked by a sample substrate frame) and that the scattering appears circular and radially symmetric.  Avoid images with streaks or non-circular polygon shaped scattering.
- Using a combination of the image and photodiode signal, pick spots with appropriate flux.  What is considered appropriate flux depends on the specific sample.  See some examples below.
    - Simply going for the highest flux (highest photodiode signal) may not always be appropriate because it may come from a region where the sample is thin or absent.
    - If the sample was floated or microtomed, regions with low photodiode signal but strong scattering may come from a region where the sample film has folded onto itself.
- If possible, pick spots that are spatially far from each other to get good representation across the sample surface.  For sample, in the scan above, images 33, 42, and 48 might be good candidates.  If using a differnet viewer in which the images are not displayed by their spatial position, keep in mind that the due to the spiral shape of the scan, images that are sequentially far apart may still be spatially close together.

It is recommended to let spiral scans run to completion, as a full spiral may help identify potential heterogeneities in a sample that might inform future sample preparation or analysis.  However, if a sufficient number of spots are found before the spiral scan is complete, the spiral scan can be aborted and then the scan queue may be resumed, which will skip the partially-complete spiral scan and move onto the next scan to save time.


## Selected desired location(s) using `nbs_viewer`

After deciding the images you want, use `nbs-viewer` to select the desired locations and update those locations in the sample configuration.
- Under the "Plot Controls" bottom left-hand pannel, click on "Load File" and select an up-to-date spreadsheet that was saved after fiducials were calibrated.
- With the desired spiral scan open, click on "Select Best Image".
- In the right-hand panel, click on the desired image(s).  The selected image(s) should have a red border after it has been clicked.
- After all desired images are selected in the scan, click "Use Selected Images" in the lower let-hand panel.
- Repeat the above steps for all spiral scans from which you want to select locations.
- Finally, click on "Save Configuration" to save out a new spreadsheet with the updated locations.
