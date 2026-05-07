const BASE_PATH = "/gutenberg";

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname === "/" || url.pathname === BASE_PATH) {
      return Response.redirect(`${url.origin}${BASE_PATH}/`, 308);
    }

    if (!url.pathname.startsWith(`${BASE_PATH}/`)) {
      return new Response("Not found", { status: 404 });
    }

    let response = await env.ASSETS.fetch(request);

    if (response.status === 404 && !url.pathname.split("/").pop().includes(".")) {
      const assetUrl = new URL(request.url);
      assetUrl.pathname = `${BASE_PATH}/index.html`;
      response = await env.ASSETS.fetch(new Request(assetUrl, request));
    }

    return response;
  },
};
