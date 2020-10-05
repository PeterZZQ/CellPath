import rpy2.robjects as robjects
# import r packages
import rpy2.robjects.packages as rpackages
import numpy as np

def principalcurve(data, thresh = 0.001, maxit = 10, stretch = 2, smoother = "smooth_spline"):
    """\
    Description
    Fits a principal curve which describes a smooth curve that passes through the middle of the data
    x in an orthogonal sense. This curve is a nonparametric generalization of a linear principal 
    component. If a closed curve is fit (using smoother = "periodic_lowess") then the starting curve 
    defaults to a circle, and each fit is followed by a bias correction suggested by Jeff Banfield.
    
    Parameters
    ----------
    x
        A matrix of points in arbitrary dimension.

    start
        Either a previously fit principal curve, or else a matrix of points that in row order define 
        a starting curve. If missing or NULL, then the first principal component is used. If the 
        smoother is "periodic_lowess", then a circle is used as the start.

    thresh
        Convergence threshold on shortest distances to the curve.

    maxit
        Maximum number of iterations.

    Stretch
        A stretch factor for the endpoints of the curve, allowing the curve to grow to avoid bunching 
        at the end. Must be a numeric value between 0 and 2.

    smoother
        choice of smoother. The default is "smooth_spline", and other choices are "lowess" and 
        "periodic_lowess". The latter allows one to fit closed curves. Beware, you may want to use 
        iter = 0 with lowess().

    approx_points
        Approximate curve after smoothing to reduce computational time. If FALSE, no approximation of 
        the curve occurs. Otherwise, approx_points must be equal to the number of points the curve gets 
        approximated to; preferably about 100.

    trace
        If TRUE, the iteration information is printed

    plot_iterations
        If TRUE the iterations are plotted.

    ...
        additional arguments to the smoothers

    s
        a parametrized curve, represented by a polygon.

    Returns
    -------
        An object `results` 

    It has components:

    projections
        a matrix corresponding to x, giving their projections onto the curve.

    order
        an index, such that s[order, ] is smooth.

    arc_len
        for each point, its arc-length from the beginning of the curve. The curve is parametrized 
        approximately by arc-length, and hence is unit-speed.

    dist
        the sum-of-squared distances from the points to their projections.

    converged
        A logical indicating whether the algorithm converged or not.

    num_itr
        Number of iterations completed before returning.
    
    """

    from rpy2.robjects import pandas2ri

    if not rpackages.isinstalled("princurve"):
        print("Installing princurve...")
        # robjects.r(
        #     """
        #     install.packages("princurve")
        #     """
        # )
        from rpy2.robjects.vectors import StrVector
        utils = rpackages.importr('utils')
        utils.chooseCRANmirror(ind=58)
        utils.install_packages(StrVector(["princurve"]))
        if rpackages.isinstalled("princurve"):
            print("\tFinished")
        else:
            print("\tInstallation failed, check lib path")
    
    pandas2ri.activate()
    princurve = rpackages.importr('princurve', on_conflict = 'warn')
    fit = princurve.principal_curve(data)
    results = {}
    results["projections"] = fit.rx2("s")
    results["order"] = fit.rx2("ord")
    results["arc_len"] = fit.rx2("lambda")
    results["dist"] = fit.rx2("dist")
    results["converged"] = fit.rx2("converged")
    results["num_itr"] = fit.rx2("num_iterations")
    
    return results
    
