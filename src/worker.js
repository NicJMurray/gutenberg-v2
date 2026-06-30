export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const assetUrl = new URL(request.url);

    if (url.pathname === "/") {
      assetUrl.pathname = "/index.html";
    }

    let response = await env.ASSETS.fetch(new Request(assetUrl, request));

    if (response.status === 404 && !url.pathname.split("/").pop().includes(".")) {
      assetUrl.pathname = "/index.html";
      response = await env.ASSETS.fetch(new Request(assetUrl, request));
    }

    return response;
  },
};
